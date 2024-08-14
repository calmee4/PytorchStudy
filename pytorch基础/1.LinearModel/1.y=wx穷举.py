# 思路 贮备一个Dataset->选择/设计model->训练（人工/自动）->推理infer
# 选择模型 先拿线性模型，效果不好拿别的模型
# 用Loss Function评估
#MSE 平均平方误差
import numpy as np
import matplotlib.pyplot as plt
import math
# 相同的索引对应一组样本
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
#  定义模型
def forward(x):
    return x*w
# 定义损失函数
def loss(x,y):
    y_pred=forward(x)
    # 求预测值和真实值的差
    return math.pow(y_pred-y,2)
# 存储w和mse大小
w_list=[]
mse_list=[]
for w in np.arange(0,4.1,0.1):
    print('w=',w)
    l_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=forward(x_val)
        loss_val=loss(x_val,y_val)
        l_sum+=loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print("MSE="," ",l_sum/3)
    print("-------------")
    w_list.append(w)
    mse_list.append(l_sum/3)
# w_list和mse_list是为了画图而存在
plt.plot(w_list,mse_list)
plt.xlabel("w_value")
plt.ylabel("MSE")
plt.show()