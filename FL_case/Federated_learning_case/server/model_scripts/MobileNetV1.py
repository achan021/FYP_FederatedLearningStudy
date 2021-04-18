#PYTORCH IMPLEMENTATION
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data import Dataset,DataLoader,TensorDataset

from tqdm import tqdm
import time
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import sklearn



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

############################################################################################################
##### Get Model
############################################################################################################
def get_net():
    model = MobileNetV1()
    return model

############################################################################################################
##### Data preprocessing function
############################################################################################################
def preprocessing(image, new_size):
    # resize to fit model
    new_img = cv2.resize(image, (new_size, new_size))
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # #median filter to remove noise and preserve edges
    # new_img = cv2.medianBlur(new_img,3)
    # #adaptive contrast stretching to stretch out the colors
    # lab = cv2.cvtColor(new_img,cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=8.0,tileGridSize=(32,32))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # new_img = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    # cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)
    return new_img

############################################################################################################
##### Load dataset
############################################################################################################
def load_dataset():

    # Parameters
    batch_size = 8

    train_img_base_path = './train_covid_folder/'
    test_img_base_path = './test_covid_folder/'

    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_img_base_path,
        transform=transform_img
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=test_img_base_path,
        transform=transform_img
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader
    return train_loader

############################################################################################################
##### Accuracy calculation function
############################################################################################################
def accuracy_cal(threshold, output, target):
    pred_output = []
    for pred in output:
        if pred >= threshold:
            pred_output.append(1)
        else:
            pred_output.append(0)

    target = target.squeeze()
    # print(target)
    target = np.array(target)
    pred_output = np.array(pred_output)
    return (target == pred_output).sum() / len(target)

############################################################################################################
##### Train model MAIN function
############################################################################################################
def train_model(model, trainloader):

    epochs = 5
    # Check for gpu use
    epoch_time = []
    train_start = time.time()
    train_losses_per_epoch = []
    train_losses_per_batch = []

    train_accuracy_per_epoch = []
    train_accuracy_per_batch = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train(trainloader, model, train_losses_per_batch, train_losses_per_epoch, train_accuracy_per_batch,
              train_accuracy_per_epoch)
        epoch_end_time = time.time()
        print('-' * 89)
        print(' end of epoch {}  time: {} - {}s '.format(epoch, epoch_end_time, epoch_start_time))
        print('-' * 89)
        epoch_time.append((epoch_end_time - epoch_start_time))
    train_end = time.time()
    full_training_time = train_end - train_start
    plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, label='train loss')
    plt.plot(range(1, len(train_accuracy_per_epoch) + 1), train_accuracy_per_epoch, label='train acc')

    plt.title('Model Result')
    plt.ylabel('result')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    with open('./mv1/training_time.txt', 'w') as os:
        os.write('Total time taken to train {} epoch : '.format(
            epochs) + '-' * 10 + '{:.10} seconds OR {:.5} Minutes\n'.format(full_training_time,
                                                                            full_training_time / 60.0))
        os.write('-' * 20 + '\n')
        os.write('EPOCH BREAKDOWN' + '\n')
        os.write('-' * 20 + '\n')
        for e, time_taken in enumerate(epoch_time):
            os.write(
                'Time taken for epoch : {}'.format(e + 1) + '-' * 10 + '{:.10} seconds OR {:.5} Minutes\n'.format(
                    time_taken, time_taken / 60.0))
            os.write('Accuracy : {}\n'.format(train_accuracy_per_epoch[e]))
            os.write('Loss : {}\n'.format(train_losses_per_epoch[e]))

############################################################################################################
##### Train model function
############################################################################################################
def train(dataloader, model, train_losses_per_batch, train_losses_per_epoch, train_accuracy_per_batch,
          train_accuracy_per_epoch):

    torch.manual_seed(24)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    print(device)

    model.to(device)

    # parameters
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    threshold = 0.5
    criterion = nn.BCEWithLogitsLoss()

    model.train()  # Turn on training mode which enables dropout.
    total_loss = 0
    cum_loss = 0
    total_acc = 0
    cum_acc = 0
    start_time = time.time()
    gradient_acc_steps = 1

    for idx, batch in tqdm(enumerate(dataloader)):

        data, targets = batch[0], batch[1]

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()

        data = data.reshape(-1, 3, 224, 224).to(device)

        # targets_GPU = targets.unsqueeze(1).to(device)
        targets_CPU = targets.unsqueeze(1).type(torch.float32).to('cpu')
        output_GPU = model(data)
        output = output_GPU.to('cpu')

        loss = criterion(output, targets_CPU)
        loss = loss / gradient_acc_steps
        loss.backward()

        if (idx % gradient_acc_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        cum_loss += loss.item()

        train_losses_per_batch.append(loss.item())
        accuracy = accuracy_cal(threshold, output, targets)
        print('batch accuracy : {}'.format(accuracy))
        train_accuracy_per_batch.append(accuracy)

        total_acc += accuracy
        cum_acc += accuracy

        if idx + 1 % 5 == 0 and idx + 1 > 0:  # display loss every intervals of 5
            cur_loss = cum_loss / 5
            cur_acc = cum_acc / 5
            # elapsed = time.time() - start_time
            print('loss : {} --- accuracy : {}'.format(cur_loss, cur_acc))
            cum_loss = 0
            cum_acc = 0
    train_losses_per_epoch.append(total_loss / len(dataloader))
    train_accuracy_per_epoch.append(total_acc / len(dataloader))

############################################################################################################
##### Evaluate model function (test)
############################################################################################################
def evaluate(model, dataloader):
    criterion = nn.BCEWithLogitsLoss()
    threshold = 0.5
    # Turn on evaluation mode which disables dropout.
    model.eval()


    torch.manual_seed(24)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model.to(device)

    total_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            data, targets = batch[0], batch[1]

            data = data.reshape(-1, 3, 224, 224).to(device)
            targets_CPU = targets.unsqueeze(1).type(torch.float32).to('cpu')
            output_GPU = model(data)
            output = output_GPU.to('cpu')

            loss = criterion(output, targets_CPU)
            total_loss += loss.item()

            accuracy = accuracy_cal(threshold, output, targets_CPU)
            total_accuracy += accuracy
    print('average acc == : {}'.format(total_accuracy / len(dataloader)))

    with open('./mv1/training_time.txt', 'a') as os:
        os.write('average testing acc == : {}\n'.format(total_accuracy / len(dataloader)))

    return (total_loss / len(dataloader)), (total_accuracy / len(dataloader))

############################################################################################################
##### Save model function
############################################################################################################
def save_model(net, PATH='../database/mobilenetv2_best.pth'):
    # torch.save(net,PATH)
    torch.save(net.state_dict(), PATH)

############################################################################################################
##### Load model function
############################################################################################################
def load_model(net, PATH='../database/mobilenetv2_best.pth'):
    # net = torch.load(PATH)
    net.load_state_dict(torch.load(PATH))
    return net