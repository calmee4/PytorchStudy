import random
import matplotlib.pyplot as plt
x_data=[1,2,3]
y_data=[2,4,6]
random.seed(42)
w=random.uniform(0,5)
def forward(x):
    return x*w
def cost(xs,ys):
    cost=0
    # 这是在算列表中的cost，针对某一个w
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost+=(y_pred-y)**2
    return cost/len(xs)
def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(w*x-y)
    return grad/len(xs)
print('Predict before training x=4 y=',forward(4))
epoch_list=[]
cost_list=[]
# 训练100 次
for epoch in range(100):
    learning_rate=0.1
    # 损失函数
    cost_val=cost(x_data,y_data)
    # gradient descent
    grad_val=gradient(x_data,y_data)
    w-=learning_rate*grad_val
    print('Epoch= ',epoch," cost= ",cost_val," w= ",w)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
plt.plot(epoch_list,cost_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()
print("After training predict x=4 y=",forward(4))
