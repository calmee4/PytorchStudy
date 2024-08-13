# 刘二大人Pytorch个人思考
<h1> Linear Model</h1>
<h2> y=w*x和w*x+b的模型</h2>
这里主要是学习如何设置自己的模型def forward(x): return w*x+b 再求损失函数 def loss(x_val,y_val): return math.pow(y_val-forward(x),2) 和MSE lose_all/sum
如果只有w 和 b的话 需要会画3维图
用matlibplot.pyplot 中的 cm和 ticker cm表示颜色，ticker 表示刻度
<h1> gradient descent</h1>
要自己学会每一步怎么梯度下降
梯度下降可以用Adam算法/SGD算法
<h1> BackPropagation </h1>
要熟练torch，比如
1。自己设定一个class (torch.nn.Module):
2. __init__():/self.forward():
3. model()种类，SGD..
4. forward()中用relu/sigmoid函数
5. epoch训练中 梯度下降怎么弄？ 
6. __init__()函数中 设置线性层linear ==>wx+b
7. TensorDataset DataLoader的用法
8. 实例化中的criterion还有optimizer 如何操作
9. epoch训练中的 optimizer.zero_grad() loss.backward() optimizer.step()
10. 画图检验
<h1>Logistic_Regression</h1>
做分类问题和上面一样，但是criterion=torch.nn.BCELoss() optimizer=.SGD
然后__init__(): self.linear=torch.nn.Linear(1,1) forward():F.sigmoid(self.linear(x)) return y_hat
<h1>Dataset，Dataloader的运用</h1>
需要用torch.nn.util中的dataset，Dataloader
dataset中需要有两个函数,__getitem__ 和__len__ ，一个是得到inputs和outputs，一个是得到数据集个数
用文件传入，再用numpy处理转为x_data和y_data
Model操作还是一样，有optim和criterion 以及loss.backward()，
<h1>全连接层对图像的操作</h1>
这里增加了一个新的库，torchvision，先导入MNIST数据集，再对这些数据进行操作，其中用到了transform，可以理解为一个操作函数，把图像ToTensor转为Input一维输入，再归一化，用标准差和平均值，之后就是正常的训练，这里虽然做的是图像处理，本质还是输入输出的逻辑回归问题
<h1>CNN操作</h1>
