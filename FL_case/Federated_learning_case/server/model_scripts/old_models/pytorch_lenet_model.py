import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.utils.data

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_net():
    net = LeNet()
    return net

def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #downlaod train dataset
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    #download test dataset
    testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=True,transform=transform)
    # types of classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes,trainset,testset

def train_model(net,trainset):
    #parameters
    epochs = 2
    batch_size = 4
    num_workers = 0
    lr = 0.001
    momentum = 0.9

    net.train()

    #create criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum = momentum)

    trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle = True,num_workers=num_workers)

    running_loss = 0.0

    for ep in range(epochs):
        for i,data in enumerate(trainloader):
            #get the inputs
            inputs_X , labels_Y = data

            #initialize the parameter gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = net(inputs_X)
            loss = criterion(outputs,labels_Y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (ep + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('finished training')
    return net

def save_model(net,PATH='../database/lenet_pytorch.pth'):
    # torch.save(net,PATH)
    torch.save(net.state_dict(),PATH)
def load_model(net,PATH='../database/lenet_pytorch.pth'):
    # net = torch.load(PATH)
    net.load_state_dict(torch.load(PATH))
    return net
def test_model(net,testset,classes):
    #hyperparameters
    batch_size = 4
    num_workers = 0

    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    net.eval()

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for data in testloader:
            inputs_X,labels_Y = data
            outputs = net(inputs_X)
            _,predicted = torch.max(outputs.data,1)
            c = (predicted == labels_Y).squeeze()
            for i in range(batch_size):
                label = labels_Y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))