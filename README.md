# 刘二大人Pytorch个人思考
<h1> Linear Model</h1>
<h2> y=w*x和w*x+b的模型
这里主要是学习如何设置自己的模型def forward(x): return w*x+b 再求损失函数 def loss(x_val,y_val): return math.pow(y_val-forward(x),2) 和MSE lose_all/sum
如果只有w 和 b的话 需要会画3维图
用matlibplot.pyplot 中的 cm和 ticker cm表示颜色，ticker 表示刻度
<h1> gradient descent</h1>
要自己学会每一步怎么梯度下降
梯度下降可以用Adam算法/SGD算法
