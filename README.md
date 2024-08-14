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
<h2>Basic</h2>
把之前的Linear层 改成了Conv2d 卷积层，还有个Subsampling，使图像大小减小 还有把数据 放在device (CUDA)中，让显卡运算，对model,inputs,outputs.to(device) 在训练的时候这么操作，还有 test的时候 with torch.no_grad()  作为测试
<h2>Advanced</h2>
设置了一个Inception（盗梦空间），把部件抽象化为Inception，进行了更加复杂的优化，效果更好，用了branch，提取了各个特征，还用了size为1的kernal（内核）减少计算，把channel变少。
<h1>RNN操作</h1>
<h2>RNNcell</h2>
需要自己手动写循环，遍历每一个SeqLen。切记，label要是long类型的，否则无法在CrossEntropy运算,然后输入input_size,hidden_size，而且只有一层，多层还得自己写
<h2>RNN</h2>
不用自己写循环，只需要input_size,hidden_size,num_layers，最终输出是一个系数矩阵，用idx.max(dim=1)取得这些行的最大值，再遍历就可以输出最终的结果。
<h2>Embedding</h2>
如果有无数个字符，那么就需要用n*n 独热编码，显然是不行的，所以用较小矩阵Embedding，不同系数代指不同值就行了
<h2>双向循环RNN</h2>
这里做了个复杂的classifier，用人名预测他的国家，分类出了18个国家，首先对人名字进行处理，转为ASCII码，再用embedding进行统筹，作为Input，中间运用了双层的 双向循环神经网络 辅助输出hidden，再用hidden经过Linear层预测出18个国家，得到最终结果。
具体操作1.国家转为了0 1 2 3 4.. 2.人名就是ASCII码 3.处理dataset，定义__len__和 __getItem__ ，把输入全部转为Tensor类型
4.对ASCII码进行排序，用排好的序列的下标取重新修正人名和国家名 使得Input和Lables一一对应 5.padded_sequence优化没有意义的0 embedding 
