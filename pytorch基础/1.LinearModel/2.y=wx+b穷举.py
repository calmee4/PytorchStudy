import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LinearLocator
import math
x_data=[1,2,3]
y_data=[2,4,6]
def forward(x):
    return w*x+b
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
w_list=[]
mse_list=[]
b_list=[]
# 我是先对应w 求的 0.1 对应b 所以w应该是x轴？
for w in np.arange(0,4.1,0.1):
    for b in np.arange(0,4.1,0.1):
        lose_all = 0
        for x_value,y_value in zip(x_data,y_data):
            lose_value=loss(x_value,y_value)
            lose_all+=lose_value
        # print('w= ',w," b= ",b," lose = ",lose_all/3)
        mse_list.append(lose_all/3)
        w_list.append(w)
        b_list.append(b)
print(mse_list)
mse_list=np.array(mse_list)
mse_list=mse_list.reshape(41,41)
mse_list=np.transpose(mse_list)
#  为什么mse_list要转至，这里把mse_list拉长成一维度，会发现mse对应的w 和b 错位了 ，所以要转至
w,b=np.meshgrid(np.unique(w_list),np.unique(b_list))
fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
surf=ax.plot_surface(w,b,mse_list,cmap=cm.coolwarm)
ax.set_zlim(0,35)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf,shrink=0.5,aspect=5)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.text2D(0.4,0.92,"Cost Values",transform=ax.transAxes)
plt.show()
# 接下来吧list转为矩阵画图
# (0,2) , (0,2)a= 0 0 1 1 b=0 1 0 15
# w=[1 2 1 2 1 2 1 2]
# b=[1 2 3 4 1 2 3 4]
