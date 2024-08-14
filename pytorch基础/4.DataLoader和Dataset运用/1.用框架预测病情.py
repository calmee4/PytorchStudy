import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 这是用来做dataset的，用于DataLoader，然后它的属性是torch的Dataset，
# 需要有两个函数 一个是输入的数（input) 也就是特征
# 一个是输出
# 第二个函数是返回子集
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 可以理解为是个操作
        # 让这层特征值全部用sigmoid转换
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1、导入数据
            inputs, labels = data
            # 2、前向传播
            y_hat = model(inputs)
            loss = criterion(y_hat, labels)
            print(epoch, i, loss.item())
            # 3、反向传播(为什么loss.backward这里不会自动补全？)
            optimizer.zero_grad()
            loss.backward()
            # 4、更新权重
            optimizer.step()

            if epoch % 30 == 1:
                y_pred_label = torch.where(y_hat >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

                accuracy = torch.eq(y_pred_label, labels).sum().item() / labels.size(0)
                print("loss = ", loss.item(), "acc = ", accuracy)

