from model_scripts import cifarnet as cfn
from model_scripts import MobileNetV1 as mv1
import os

def main(model_type):
    if model_type == "mv1":
        model = mv1.get_net()
        os.makedirs('./mv1',exist_ok=True)
        train_loader, test_loader = mv1.load_dataset()
        # model = mv1.load_model(model,'./mv1/train_70_e5.pth')
        mv1.train_model(model, train_loader)
        mv1.save_model(model, '../database/mobilenetv1_global.pth')
        mv1.evaluate(model, test_loader)

    elif model_type == "cifarnet":
        model = cfn.get_net()
        os.makedirs('./cfn', exist_ok=True)
        train_loader, test_loader = cfn.load_dataset()
        # model = cfn.load_model(model, './cfn/train_70_e5.pth')
        cfn.train_model(model, train_loader)
        cfn.save_model(model, '../database/cifarnet_global.pth')
        cfn.evaluate(model, test_loader)

    # model_urls = {
    #     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    # }
    # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                       progress=True)
    #
    # load_para(model,state_dict)
    # model = mmv2.load_model(mmv2.get_net(), './base_model_manual/train_70_e10_lr0003.pth')

    # print(model.pretrainedModel.classifier)
    # mmv2.save_model(model, './base_model_manual/train_70_e20_lr0003.pth')
    # #


# def load_para(model,state_dict):
#     try:
#         model.load_state_dict(state_dict,strict=False)
#     except RuntimeError:
#         print(RuntimeError)

model_type = "cifarnet"
main(model_type)