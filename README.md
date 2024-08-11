# 刘二大人Pytorch个人思考
<h1> Linear Model
<h2> y=w*x和w*x+b的模型
这里主要是学习如何设置自己的模型def forward(x): return w*x+b 再求损失函数 def loss(x_val,y_val): return math.pow(y_val-forward(x),2) 和MSE lose_all/sum
