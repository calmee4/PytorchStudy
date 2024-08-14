import random
import matplotlib.pyplot as plt
x_data=[1,2,3]
y_data=[2,4,6]
random.seed(42)
w=random.uniform(0,5)

def forward(x):
    return w*x
print("训练前 x=4 y=",forward(4))
def cost(xs,ys):
    y_pred=forward(xs)
    return (y_pred-ys)**2
def gradient(x,y):
    return 2*x*(w*x-y)
epoch_list=[]
cost_list=[]
for epoch in range(0,100):
    cost_all=0
    # 这里计算每一个epoch的cost
    for x_value,y_value in zip(x_data,y_data):
        cost_all+=cost(x_value,y_value)
    index=random.randint(0,len(x_data)-1)
    x_rand=x_data[index]
    y_rand=y_data[index]
    learning_rate=0.01
    w-=learning_rate*gradient(x_rand,y_rand)
    epoch_list.append(epoch)
    cost_list.append(cost_all)
    print("epoch= ",epoch," w= ",w," cost= ",cost_all)
plt.plot(epoch_list,cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()
print("训练后 x=4 y=",forward(4))