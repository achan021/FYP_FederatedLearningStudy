#PYTORCH IMPLEMENTATION
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pandas as pd
from torch.utils.data import Dataset,DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os

############################################################################################################
##### Model class declaration
############################################################################################################
class mobilenetv2Model(nn.Module):
  def __init__(self):
    super(mobilenetv2Model,self).__init__()

    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = 1280 #cannot get from classifier component
    model.classifier = nn.Sequential(nn.Dropout(p=0.2,inplace=False),
                                          nn.Linear(in_features = num_ftrs,out_features = 256),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(256,1))
    self.pretrainedModel = model
  def forward(self,x):
    x = self.pretrainedModel(x)
    return x

############################################################################################################
##### Get Model
############################################################################################################
def get_net():
    model = mobilenetv2Model()
    return model

############################################################################################################
##### Data preprocessing function
############################################################################################################
def preprocessing(image,new_size):
    #resize to fit model
    new_img = cv2.resize(image,(new_size,new_size))
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

    #Parameters
    batch_size = 16

    #Data is in covid_images
    image_base_path = './covid_images/'
    train_df = pd.read_csv('./train_df_1.csv')
    test_df = pd.read_csv('./test_df.csv')

    x_train = train_df['filename']
    y_train = train_df['finding']
    x_test = test_df['filename']
    y_test = test_df['finding']

    train_covid_partition_folder = './train_covid_folder/'
    train_covid_label_1 = train_covid_partition_folder + str(train_df['finding'].unique()[0]) + '/'
    train_covid_label_2 = train_covid_partition_folder + str(train_df['finding'].unique()[1]) + '/'

    test_covid_partition_folder = './test_covid_folder/'
    test_covid_label_1 = test_covid_partition_folder + str(test_df['finding'].unique()[0]) + '/'
    test_covid_label_2 = test_covid_partition_folder + str(test_df['finding'].unique()[1]) + '/'

    if not os.path.isdir(train_covid_partition_folder):
        os.mkdir(train_covid_partition_folder)
        if not os.path.isdir(train_covid_label_1):
            os.mkdir(train_covid_label_1)
        if not os.path.isdir(train_covid_label_2):
            os.mkdir(train_covid_label_2)

        # Partiton images into respective folders
        for filename, cat in tqdm(zip(x_train, y_train)):
            img = cv2.imread(image_base_path + filename)
            # img = preprocessing(img, 233)
            # img = img / 255
            if cat == train_df['finding'].unique()[0]:
                cv2.imwrite(train_covid_label_1 + filename, img)
            elif cat == train_df['finding'].unique()[1]:
                cv2.imwrite(train_covid_label_2 + filename, img)

    if not os.path.isdir(test_covid_partition_folder):
        os.mkdir(test_covid_partition_folder)
        if not os.path.isdir(test_covid_label_1):
            os.mkdir(test_covid_label_1)
        if not os.path.isdir(test_covid_label_2):
            os.mkdir(test_covid_label_2)
        for filename, cat in tqdm(zip(x_test, y_test)):
            img = cv2.imread(image_base_path + filename)
            # img = preprocessing(img, 233)
            # img = img / 255
            if cat == train_df['finding'].unique()[0]:
                cv2.imwrite(test_covid_label_1 + filename, img)
            elif cat == train_df['finding'].unique()[1]:
                cv2.imwrite(test_covid_label_2 + filename, img)




    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root = train_covid_partition_folder,
        transform=transform_img
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=test_covid_partition_folder,
        transform=transform_img
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # x_train = train_df['filename']
    # y_train = train_df['finding']
    # x_test = test_df['filename']
    # y_test = test_df['finding']
    #
    # #Load image data into CPU memory
    # x_batch_img = []
    # y_batch_train = []
    # for filename, cat in tqdm(zip(x_train, y_train)):
    #     img = cv2.imread(image_base_path + filename)
    #     img = preprocessing(img, 233)
    #     img = img / 255
    #     x_batch_img.append(img)
    #     y_batch_train.append(cat)
    # y_batch_train = np.array(y_batch_train)
    # x_batch_img_train = np.stack(x_batch_img, axis=0)
    #
    # x_batch_img = []
    # y_batch_test = []
    # for filename, cat in tqdm(zip(x_test, y_test)):
    #     img = cv2.imread(image_base_path + filename)
    #     img = preprocessing(img, 233)
    #     img = img / 255
    #     x_batch_img.append(img)
    #     y_batch_test.append(cat)
    # y_batch_test = np.array(y_batch_test)
    # x_batch_img_test = np.stack(x_batch_img, axis=0)

    # x_train_torch = torch.tensor(x_batch_img_train).type(torch.float32)
    # y_train_torch = torch.tensor(y_batch_train).type(torch.float32)
    #
    # x_test_torch = torch.tensor(x_batch_img_test).type(torch.float32)
    # y_test_torch = torch.tensor(y_batch_test).type(torch.float32)
    #
    # # Training Dataloaders
    # dataset = TensorDataset(x_train_torch, y_train_torch)
    # train_loader = DataLoader(dataset, batch_size=batch_size)
    #
    # # Testing Dataloaders
    # dataset = TensorDataset(x_test_torch, y_test_torch)
    # test_loader = DataLoader(dataset, batch_size=batch_size)

    return train_loader,test_loader

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
    return accuracy_score(target, pred_output)

############################################################################################################
##### Train model MAIN function
############################################################################################################
def train_model(model,trainloader):

    epochs = 5
    # Check for gpu use

    train_losses_per_epoch = []
    train_losses_per_batch = []

    train_accuracy_per_epoch = []
    train_accuracy_per_batch = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train(trainloader, model,train_losses_per_batch,train_losses_per_epoch,train_accuracy_per_batch,train_accuracy_per_epoch)
        print('-' * 89)
        print(' end of epoch {}  time: {} - {}s '.format(epoch,time.time(),epoch_start_time))
        print('-' * 89)
    plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, label='train loss')
    plt.plot(range(1, len(train_accuracy_per_epoch) + 1), train_accuracy_per_epoch, label='train acc')

    plt.title('Model Result')
    plt.ylabel('result')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()



############################################################################################################
##### Train model function
############################################################################################################
def train(dataloader, model,train_losses_per_batch,train_losses_per_epoch,train_accuracy_per_batch,train_accuracy_per_epoch):

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


    model.train() # Turn on training mode which enables dropout.
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

        data = data.reshape(-1, 3, 233, 233).to(device)

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

        if idx+1 % 5 == 0 and idx+1 > 0:  # display loss every intervals of 5
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
def evaluate(model,dataloader):
    criterion = nn.BCEWithLogitsLoss()
    threshold = 0.5
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            data, targets = batch[0], batch[1]

            data = data.reshape(-1, 3, 233, 233)
            targets = targets.unsqueeze(1).type(torch.float32)

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            accuracy = accuracy_cal(threshold, outputs, targets)
            total_accuracy += accuracy
    print('average acc == : {}'.format(total_accuracy / len(dataloader)))

    return (total_loss / len(dataloader)) , (total_accuracy / len(dataloader))

############################################################################################################
##### Save model function
############################################################################################################
def save_model(net,PATH='../database/mobilenetv2_best.pth'):
    # torch.save(net,PATH)
    torch.save(net.state_dict(),PATH)

############################################################################################################
##### Load model function
############################################################################################################
def load_model(net,PATH='../database/mobilenetv2_best.pth'):
    # net = torch.load(PATH)
    net.load_state_dict(torch.load(PATH))
    return net