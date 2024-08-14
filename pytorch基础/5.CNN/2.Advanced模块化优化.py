# 为了减少代码冗余所以直接封装成类
# 用了多种卷积核，最终合并，得到多个特征，很像随机森林
# 最终Concatenate 拼接，得到一致的大小，但是Conv后的大小不一样，所以要padding
# 还有个ResNet，是为了得残差residual
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets  # dataset 引用位置
from torch.utils.data import DataLoader  # DataLoader 引用位置
from torchvision import transforms
batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
class InceptionA(torch.nn.Module):
    def __init__(self,in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 压缩
        # 16
        self.branch1x1=nn.Conv2d(in_channels,16,kernel_size=1)
        # 24
        self.branch5x5_1=nn.Conv2d(in_channels,16,kernel_size=1)
        # 用的kernel_size 使得图像长宽高都少了4，所以padding=2 扩展4格
        self.branch5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2)
        # 24
        self.branch3x3_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=nn.Conv2d(24,24,kernel_size=3,padding=1)
        # 24        
        self.branch_pool=nn.Conv2d(in_channels,24,kernel_size=1)
        
    def forward(self,x):
        branch1x1=self.branch1x1(x)
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)
        output=[branch1x1,branch5x5,branch3x3,branch_pool]
        return torch.cat(output,dim=1)
class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 其实是一个大概
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)
        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)
        self.mp=nn.MaxPool2d(2)
        # 88
        self.fc=nn.Linear(1408,10)
    def forward(self,x):
        # batch_size
        in_size=x.size(0)
        # -4 /2  12
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        # -4 /2 4
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x
model=Net()
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
def train(epoch):
    running_loss=0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch_idx%300==299:
            print(f"轮数={epoch}时 loss={running_loss/300}")
            running_loss=0
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,labels=data
            outputs=model(inputs)
            _,pred=torch.max(outputs,dim=1)
            total+=labels.size(0)
            correct+=(pred==labels).sum().item()
        print(f"Accuracy: {correct/total*100}")
        return correct/total*100
import matplotlib.pyplot as plt
if __name__ =="__main__":
    epoch_list=[]
    loss_list=[]
    for epoch in range(10):
        train(epoch)
        loss_list.append(test())
        epoch_list.append(epoch)
    plt.plot(epoch_list,loss_list)
    plt.show()