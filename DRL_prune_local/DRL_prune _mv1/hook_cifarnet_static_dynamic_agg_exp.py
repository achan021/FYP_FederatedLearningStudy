import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torchvision

def return_none(*args, **kwargs):  
    return None
def identity(x):
    return x
def return_ones(x):
    x, downsampled, gates, static_gate = x
    with torch.no_grad():
        return torch.ones_like(gates), torch.ones_like(gates), 1.

class GatedConv(nn.Module):
    def __init__(self, conv_bn_relu, 
            predict_gate_with_filter_dist):
        super(GatedConv, self).__init__()
        self.conv_bn_relu = conv_bn_relu

        #Note that convBNReLU == Conv2d + batchnorm + relu
        out_channels = self.conv_bn_relu.conv.out_channels
        in_channels = self.conv_bn_relu.conv.in_channels

        self.gate = nn.Linear(in_channels,out_channels)

        self.static_gate = nn.Parameter(torch.zeros(out_channels))
        self.predict_gate_with_filter_dist = predict_gate_with_filter_dist

        self.gate_beta = nn.Parameter(torch.zeros(out_channels))

        self.gate.bias.data.fill_(0.)
        self.predict_gate_with_filter_dist[-1].bias.data.fill_(1.)

        self.filter_dist = nn.Parameter(torch.zeros(out_channels))
        self.filter_dist.requires_grad = False
        self.filter_norm = nn.Parameter(torch.zeros(out_channels))
        self.filter_norm.requires_grad = False

        self.taylor_first = nn.Parameter(torch.zeros(out_channels))
        self.taylor_first.requires_grad = False

        self.copy_bn_weight_to_gate_bias_and_beta()

    def forward(self, x, moduel_hook1=return_none, moduel_hook2=return_none, pruning_hook1=return_ones):
        #-----------------runtime importance predictor-----------------------
        #self.gate = nn.Linear(self.in_channels,self.out_channels)
        downsampled = F.avg_pool2d(x, (x.shape[2],x.shape[2]))
        # downsampled = F.avg_pool2d(x, x.shape[2])
        downsampled = downsampled.view(x.shape[0], x.shape[1])
        gates = self.gate(downsampled)


        #-----------------static importance predictor------------------------
        # self.predict_gate_with_filter_dist = nn.Sequential(
        #     nn.Linear(4, 100),
        #     nn.SELU(),
        #     nn.Linear(100, 1)
        # )

        filter_dist = self.filter_dist[..., None]
        filter_norm = self.filter_norm[..., None]

        static_gates = self.static_gate.clone() #learnable parameter.
        taylor_first = self.taylor_first[..., None]
        static_gates = self.predict_gate_with_filter_dist(
            torch.cat(
                (
                    static_gates[..., None], 
                    filter_dist, 
                    filter_norm, 
                    taylor_first,
                ),
                dim=1
            )
        ) # [out_channel, 1]
        static_gates = static_gates.squeeze(dim=1).unsqueeze(dim=0) # [1, out_channel]
        
        sign_gates = torch.sign(gates)
        sign_static_gates = torch.sign(static_gates)
        negative_negative = (1. - sign_gates) * (1. - sign_static_gates) / 4.
        negative_negative_to_negative = 1. - negative_negative * 2.
        negative_negative_to_negative = negative_negative_to_negative.detach()

        moduel_hook2(self.conv_bn_relu.conv)

        x = self.conv_bn_relu.conv(x.to('cuda'))
        x = self.conv_bn_relu.bn(x)


        active, static_active, budget = pruning_hook1((x, downsampled, gates, static_gates))

        static_gates = static_gates * static_active
        gates = negative_negative_to_negative * gates * static_gates
        gates = gates * active
        gates = gates[..., None, None]
        beta = self.gate_beta[None, ...] * active * static_active

        x = x * gates + beta[..., None, None]

        x = nn.ReLU(inplace=True)(x)

        active_ratio = (active * static_active).mean()
        moduel_hook1(x)

        return x, active_ratio

    def copy_bn_weight_to_gate_bias_and_beta(self):
        batch_norm_layer = self.conv_bn_relu.bn

        if self.gate.bias.shape != batch_norm_layer.weight.shape :
            raise Exception('shape mismatch: self.gate.bias.shape {} vs batch_norm_layer.weight.shape {}'.format(
                self.gate.bias.shape,
                batch_norm_layer.weight.shape,
            ))
        if self.gate_beta.shape != batch_norm_layer.weight.shape :
            raise Exception('shape mismatch: self.gate_beta.shape {} vs batch_norm_layer.weight.shape {}'.format(
                self.gate_beta.shape,
                batch_norm_layer.weight.shape,
            ))
        # copy bn weight to gate bias
        self.gate.bias.data.copy_(batch_norm_layer.weight.data)
        batch_norm_layer.weight.requires_grad = False
        batch_norm_layer.weight.data.fill_(1.)
        # copy bn bias to beta
        self.gate_beta.data.copy_(batch_norm_layer.bias.data)
        batch_norm_layer.bias.requires_grad = False
        batch_norm_layer.bias.data.fill_(0.)


    def update_taylor_first(self, n_data_point:int):
        conv_layer = self.conv_bn_relu.bn
        taylor_first = conv_layer.weight.data * conv_layer.weight.grad
        taylor_first = taylor_first.div_(n_data_point).detach()
        taylor_first = taylor_first.pow(2.).sum(dim=(1, 2, 3))
        taylor_first = taylor_first / (taylor_first.norm().item() + 1e-3)
        self.taylor_first.copy_(taylor_first)

    def fbs_parameters(self):
        for param in self.gate.parameters():
            yield param
        yield self.gate_beta
        for param in self.static_gate_parameters():
            yield param

    def static_gate_parameters(self):
        yield self.static_gate
        for param in self.predict_gate_with_filter_dist.parameters():
            yield param

    def gate_parameters(self):
        for param in self.gate.parameters():
            yield param

class PreserveSignMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, static_gates):
        sign = ((torch.sign(gates) + 1.) / 2.) * ((torch.sign(static_gates) + 1.) / 2.)
        sign = sign * 2 - 1.
        ctx.save_for_backward(gates, static_gates, sign)
        return sign * gates.abs() * static_gates.abs()
    
    @staticmethod
    def backward(ctx, grad_output):
        gates, static_gates, sign = ctx.saved_tensors
        return grad_output * static_gates, grad_output * gates

class MultiplyMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, active):
        result = gates * active
        ctx.save_for_backward(result, gates, active)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, gates, active = ctx.saved_tensors
        return grad_output, grad_output * gates








class conv_bn(nn.Module):
  def __init__(self,inp, oup, stride):
    super(conv_bn,self).__init__()
    self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
    self.bn =  nn.BatchNorm2d(oup)
  def forward(self,x):
    x = self.conv(x)
    x = self.bn(x)
    x = nn.ReLU(inplace=True)(x)
    return x

class conv_dw(nn.Module):
  def __init__(self,inp, oup, stride):
    super(conv_dw,self).__init__()
    self.conv = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
    self.bn =  nn.BatchNorm2d(inp)
  def forward(self,x):
    x = self.conv(x)
    x = self.bn(x)
    x = nn.ReLU(inplace=True)(x)
    return x

class conv_pw(nn.Module):
  def __init__(self,inp, oup, stride):
    super(conv_pw,self).__init__()
    self.conv =  nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
    self.bn =  nn.BatchNorm2d(oup)
  def forward(self,x):
    x = self.conv(x)
    x = self.bn(x)
    x = nn.ReLU(inplace=True)(x)
    return x

class GatedMv1(nn.Module):
    def __init__(self,mv1_net):
        super(GatedMv1,self).__init__()
        self.predict_gate_with_filter_dist = nn.Sequential(
            nn.Linear(4, 100),
            nn.SELU(),
            nn.Linear(100, 1)
        )
        self.mv1_net = mv1_net

        self.gconv0 = GatedConv(self.mv1_net.conv0, self.predict_gate_with_filter_dist)
        self.gconv1 = GatedConv(self.mv1_net.conv1, self.predict_gate_with_filter_dist)
        self.gconv2 = GatedConv(self.mv1_net.conv2, self.predict_gate_with_filter_dist)

        self.gconv3 = GatedConv(self.mv1_net.conv3, self.predict_gate_with_filter_dist)
        self.gconv4 = GatedConv(self.mv1_net.conv4, self.predict_gate_with_filter_dist)

        self.gconv5 = GatedConv(self.mv1_net.conv5, self.predict_gate_with_filter_dist)
        self.gconv6 = GatedConv(self.mv1_net.conv6, self.predict_gate_with_filter_dist)

        self.gconv7 = GatedConv(self.mv1_net.conv7, self.predict_gate_with_filter_dist)
        self.gconv8 = GatedConv(self.mv1_net.conv8, self.predict_gate_with_filter_dist)

        self.gconv9 = GatedConv(self.mv1_net.conv9, self.predict_gate_with_filter_dist)
        self.gconv10 = GatedConv(self.mv1_net.conv10, self.predict_gate_with_filter_dist)

        self.gconv11 = GatedConv(self.mv1_net.conv11, self.predict_gate_with_filter_dist)
        self.gconv12 = GatedConv(self.mv1_net.conv12, self.predict_gate_with_filter_dist)

        self.gconv13 = GatedConv(self.mv1_net.conv13, self.predict_gate_with_filter_dist)
        self.gconv14 = GatedConv(self.mv1_net.conv14, self.predict_gate_with_filter_dist)

        self.gconv15 = GatedConv(self.mv1_net.conv15, self.predict_gate_with_filter_dist)
        self.gconv16 = GatedConv(self.mv1_net.conv16, self.predict_gate_with_filter_dist)

        self.gconv17 = GatedConv(self.mv1_net.conv17, self.predict_gate_with_filter_dist)
        self.gconv18 = GatedConv(self.mv1_net.conv18, self.predict_gate_with_filter_dist)

        self.gconv19 = GatedConv(self.mv1_net.conv19, self.predict_gate_with_filter_dist)
        self.gconv20 = GatedConv(self.mv1_net.conv20, self.predict_gate_with_filter_dist)

        self.gconv21 = GatedConv(self.mv1_net.conv21, self.predict_gate_with_filter_dist)
        self.gconv22 = GatedConv(self.mv1_net.conv22, self.predict_gate_with_filter_dist)

        self.gconv23 = GatedConv(self.mv1_net.conv23, self.predict_gate_with_filter_dist)
        self.gconv24 = GatedConv(self.mv1_net.conv24, self.predict_gate_with_filter_dist)

        self.gconv25 = GatedConv(self.mv1_net.conv25, self.predict_gate_with_filter_dist)
        self.gconv26 = GatedConv(self.mv1_net.conv26, self.predict_gate_with_filter_dist)

        self.fc = self.mv1_net.fc

    def forward(self, x, moduel_hook1=return_none, moduel_hook2=return_none, pruning_hook1=return_ones):

        xs = []
        lassos = []
        x, lasso = self.gconv0(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv1(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv2(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv3(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv4(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv5(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv6(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv7(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv8(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv9(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv10(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv11(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv12(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv13(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv14(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv15(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv16(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv17(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv18(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv19(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv20(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv21(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv22(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv23(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv24(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv25(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv26(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)

        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        lassos = torch.stack(lassos, dim=-1)
        return  x, lassos



class MobileNetV1(nn.Module):
    def __init__(self, ch_in=3, n_classes=1):
        super(MobileNetV1, self).__init__()

        self.conv0 = conv_bn(ch_in, 32, 2)
        self.conv1 = conv_dw(32, 64, 1)
        self.conv2 = conv_pw(32, 64, 1)

        self.conv3 = conv_dw(64, 128, 2)
        self.conv4 = conv_pw(64, 128, 2)

        self.conv5 = conv_dw(128, 128, 1)
        self.conv6 = conv_pw(128, 128, 1)

        self.conv7 = conv_dw(128, 256, 2)
        self.conv8 = conv_pw(128, 256, 2)

        self.conv9 = conv_dw(256, 256, 1)
        self.conv10 = conv_pw(256, 256, 1)

        self.conv11 = conv_dw(256, 512, 2)
        self.conv12 = conv_pw(256, 512, 2)

        self.conv13 = conv_dw(512, 512, 1)
        self.conv14 = conv_pw(512, 512, 1)

        self.conv15 = conv_dw(512, 512, 1)
        self.conv16 = conv_pw(512, 512, 1)

        self.conv17 = conv_dw(512, 512, 1)
        self.conv18 = conv_pw(512, 512, 1)

        self.conv19 = conv_dw(512, 512, 1)
        self.conv20 = conv_pw(512, 512, 1)

        self.conv21 = conv_dw(512, 512, 1)
        self.conv22 = conv_pw(512, 512, 1)

        self.conv23 = conv_dw(512, 1024, 2)
        self.conv24 = conv_pw(512, 1024, 2)

        self.conv25 = conv_dw(1024, 1024, 1)
        self.conv26 = conv_pw(1024, 1024, 1)

        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)
        x = self.conv26(x)

        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



    def update_taylor_first(self, n_data_point):
        for c in self.children():
            if hasattr(type(c), 'update_taylor_first'):
                c.update_taylor_first(n_data_point)

    def basenet_parameters(self):
        return self.mobilenetv2Model.parameters()

    def fbs_parameters(self):
        for name, param in self.named_parameters():
            if 'gate' in name and 'gconv' in name:  # and ('gate_bn' not in name):
                yield param
        # gconv0.gate.weight
        # gconv0.gate.bias
        # gconv0.gate_bn.weight
        # gconv0.gate_bn.bias
        # gconv1.gate.weight
        # gconv1.gate.bias
        # gconv1.gate_bn.weight
        # gconv1.gate_bn.bias
        # gconv2.gate.weight
        # gconv2.gate.bias
        # gconv2.gate_bn.weight
        # gconv2.gate_bn.bias
        # gconv3.gate.weight
        # gconv3.gate.bias
        # gconv3.gate_bn.weight
        # gconv3.gate_bn.bias
        # gconv4.gate.weight
        # gconv4.gate.bias
        # gconv4.gate_bn.weight
        # gconv4.gate_bn.bias
        # gconv5.gate.weight
        # gconv5.gate.bias
        # gconv5.gate_bn.weight
        # gconv5.gate_bn.bias
        # gconv6.gate.weight
        # gconv6.gate.bias
        # gconv6.gate_bn.weight
        # gconv6.gate_bn.bias
        # gconv7.gate.weight
        # gconv7.gate.bias
        # gconv7.gate_bn.weight
        # gconv7.gate_bn.bias

    @staticmethod
    def compute_loss(outputs, targets, lassos):
        loss = F.cross_entropy(outputs, targets)
        lassos_loss = 1e-8 * lassos.sum()
        loss = loss + lassos_loss
        return loss, lassos_loss



c = MobileNetV1()
pass