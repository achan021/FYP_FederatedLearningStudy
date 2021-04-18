import asyncio
import websockets
import pickle
import torch
import sys
import os
import math

from model_scripts import pytorch_inception_model as pi


PATH_rpi = '../database/inception_pytorch_rpi.pth'
PATH_original = '../database/inceptionv3_best.pth'

model_rpi = pi.get_net()
model_original = pi.get_net()

model_rpi.load_state_dict(torch.load(PATH_rpi))
model_original.load_state_dict(torch.load(PATH_original))

train_loader, test_loader = pi.load_dataset()

org_loss,org_acc = pi.evaluate(model_original,test_loader)
rpi_loss,rpi_acc = pi.evaluate(model_rpi,test_loader)

print('Original model evaluation : loss : {} ----- acc : {} '.format(org_loss,org_acc))
print('RPI model evaluation : loss : {} ----- acc : {} '.format(rpi_loss,rpi_acc))

sdRpi = model_rpi.state_dict()
sdOrg = model_original.state_dict()

for key in sdOrg:
    sdOrg[key] = (sdOrg[key] + sdRpi[key]) / 2

model_original.load_state_dict(sdOrg)
agg_loss,agg_acc = pi.evaluate(model_original,test_loader)
print('AGG model evaluation : loss : {} ----- acc : {} '.format(agg_loss,agg_acc))