#import torchvision
import torch
from torchvision import datasets,transforms
from torch.autograd import Variable
#import torch.optim as optim
import time
transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,))])

train_data=datasets.MNIST(root=".//data//",
                          transform=transform,
                          train=True,
                          download=True)
test_data=datasets.MNIST(".//data//",
                         transform=transform,
                         train=False,
                         download=True)

train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=64,
                                              shuffle=True)
test_data_loader=torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=64,
                                              shuffle=True)

num_i=28*28
num_h=100
num_o=10
batch_size=64


class Model(torch.nn.Module):
    def __init__(self,num_i,num_h,num_o):
        super(Model, self).__init__()

        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_o)

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu2(x)
        x=self.linear3(x)
        return x
model=Model(num_i,num_h,num_o)
cost=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())
epochs=10

for epoch in range(epochs):
    sum_loss=0
    train_correct=0
    for data in  train_data_loader:
        inputs,labels=data
        inputs=torch.flatten(inputs,start_dim=1)
        outputs=model(inputs)
        optimizer.zero_grad()
        loss=cost(outputs,labels)
        loss.backward()
        optimizer.step()
        _,id=torch.max(outputs.data,1)
        sum_loss+=loss.data
        train_correct+=torch.sum(id==labels.data)
        print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(train_data_loader)))
        print('        correct:%.03f%%' % (100 * train_correct / len(train_data)))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
model.eval()
test_correct=0
for data in test_data_loader:
    inputs,labels=data
    inputs,labels=Variable(inputs).cpu(),Variable(labels).cpu()
    inputs=torch.flatten(inputs,start_dim=1)
    outputs=model(inputs)
    _,id=torch.max(outputs.data,1)
    test_correct+=torch.sum(id==labels.data)
print("correct:%.3f%%"%(100*test_correct/len(test_data)))
















