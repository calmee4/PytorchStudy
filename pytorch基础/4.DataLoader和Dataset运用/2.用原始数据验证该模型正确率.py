import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,random_split
# import torch
import csv
class DiabetesDataset(Dataset):
    def __init__(self,filepath) -> None:
        super().__init__()
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        # 第一维的长度
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
dataset=DiabetesDataset('diabetes.csv')
train_size=int(0.9*len(dataset))
test_size=len(dataset)-train_size
train_dataset,test_dataset=random_split(dataset,[train_size,test_size])
train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=4)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=True,num_workers=4)
class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
outputcsv='training.csv'
if __name__=='__main__':
    with open(outputcsv,mode='w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['epoch','loss'])
        for epoch in range(10):
            for i ,data in enumerate(train_loader,0):
                inputs,outputs=data
                pred=model(inputs)
                loss=criterion(pred,outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"第{epoch}轮已经完成")
    torch.save(model.state_dict(),'diabetes_model.pth')
    print("模型已保存")
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
     for data in test_loader:
        inputs,labels=data
        outputs=model(inputs)
        predicted=(outputs>0.5).float()
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=correct/total
    print("准确率为  ",accuracy)
    content=torch.load('diabetes_model.pth')
    print("参数")
    print(content.keys())
    print(content['model'])
    