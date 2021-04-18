import os
import time
import argparse
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset

import server.model_scripts.DRL_cifarnet.hook_cifarnet_static_dynamic_agg as hook_cifarnet
from server.model_scripts.DRL_cifarnet.BudgetPruning_static_dynamic_exp import PruningInference
from server.model_scripts.DRL_cifarnet.PPOAgentGaussian_exp import GaussianActorCritic
from server.model_scripts.DRL_cifarnet.PPOAgentGaussian_static_exp import GaussianActorCritic_static

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--budget', type=float, default=0.5) #static budget (budget)
parser.add_argument('--dpratio', type=float, default=0.5) #dynamic prune ratio (dpratio)
program_args = parser.parse_args()
print('budget ratio : {} -------- dynamic pruning ratio : {}'.format(program_args.budget,program_args.dpratio))

arg = {}
arg['sigma'] = 0.02 #gaussian param (SD)
arg['budget'] = program_args.budget
arg['static_budget'] = arg['budget']
arg['dynamic_prune_ratio'] = program_args.dpratio

#Data is in covid_images
batch_size = 8
train_img_base_path = './train_covid_folder/'

def load_dataset():

    transform_img = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root = train_img_base_path,
        transform=transform_img
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader


def train_basenet(pruning_net, train_loader, optimizer,mean_sparsity_state_dict,train_save_path):
    epochs = 5

    global cur_batch_win
    pruning_net.base_net.train()
    pruning_net.actor.eval()


    criterion = nn.BCEWithLogitsLoss()

    loss_list, batch_list, r_list, a_list = [], [], [], []

    for e in range(epochs):
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            images = images
            labels = labels
            loss = 0.0

            sample_budget = torch.ones([images.shape[0], 1]).to(images.device) * arg['budget']
            static_budget = torch.ones([images.shape[0], 1]).to(images.device) * arg['static_budget']

            output, _, actions_list, _, static_info = pruning_net(images, sample_budget, static_budget,is_train_gagent=False, is_train_base=True)
            output, _ = output

            labels = labels.unsqueeze(1).type(torch.float32)

            loss = criterion(output, labels)

            pred_output = []
            batch_correct = 0
            threshold = 0.5
            for i_pred in output:
                if i_pred >= threshold:
                    pred_output.append(1)
                else:
                    pred_output.append(0)

            pred_output = torch.FloatTensor(pred_output)
            batch_correct += pred_output.eq(labels.view_as(pred_output)).sum().tolist()

            loss_list.append(loss.clone().detach().cpu().item())
            batch_list.append(i + 1)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(pruning_net.base_net.parameters(), 40.)
            optimizer.step()

        print('saving model after epoch...')
        torch.save(
            {
                'pruning_net': pruning_net.state_dict(),
                'actor_optimizer': pruning_net.actor.state_dict(),
                'mean_sparsity': mean_sparsity_state_dict,
                'basenet_optimizer': optimizer.state_dict()
            },
            train_save_path
        )
        print('Train BaseNet - Epoch {} complete. loss: {:.6g}'.format(
            e+1,
            torch.tensor(loss_list).mean(),
        ))

def get_net():
    base_net = hook_cifarnet.CifarNet()
    net = hook_cifarnet.GatedCifarNet(base_net)

    pruning_net = PruningInference(base_net=net)  # wrap the pruning net into a pruning inference class
    pruning_net.init_basenet(input_shape=(3, 32, 32))

    # setting up the runtime agent
    gaussian_actor = GaussianActorCritic(rnn_hidden_size=128,
                                         prunable_layers_n_channels=pruning_net.layers_n_channels,
                                         sigma=arg['sigma'], )
    # setting up the static agent
    static_actor = GaussianActorCritic_static(rnn_hidden_size=128,
                                              prunable_layers_n_channels=pruning_net.layers_n_channels,
                                              sigma=arg['sigma'], )

    pruning_net.actor = gaussian_actor
    pruning_net.static_actor = static_actor



    return pruning_net

def load_model(pruning_net,model_save_path='./saved_ppo_static_dynamic_model/cifarnet_budget_0.5_dpratio_0.5_epoch_5_budget_0.5.pth'):
    savefile = torch.load(model_save_path,
                          map_location=torch.device('cpu'))

    pruning_net.load_state_dict(savefile['pruning_net'])

    pruning_net.dynamic_prune_ratio = arg['dynamic_prune_ratio']
    optimizer = optim.Adam(pruning_net.base_net.parameters(), lr=0.0005)
    optimizer.load_state_dict(savefile['basenet_optimizer'])
    mean_sparsity_state_dict = savefile['mean_sparsity']

    return pruning_net,optimizer,mean_sparsity_state_dict

if __name__ == "__main__":

    train_loader = load_dataset()
    model = get_net()
    model, optimizer,mean_sparsity_state_dict = load_model(model)


    save_path = './saved_ppo_static_dynamic_model/client_trained_cifarnet_budget_0.5_dpratio_0.5_epoch_5_budget_0.5.pth'

    train_basenet(model,train_loader,optimizer,mean_sparsity_state_dict,save_path)






