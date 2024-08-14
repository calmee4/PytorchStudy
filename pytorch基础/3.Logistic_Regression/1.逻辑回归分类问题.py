import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_hat = F.sigmoid(self.linear(x))
        return y_hat


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

for epoch in range(1000):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_list.append(epoch)
    loss_list.append(loss.item())

print('w =', model.linear.weight.item())
print('b =', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

