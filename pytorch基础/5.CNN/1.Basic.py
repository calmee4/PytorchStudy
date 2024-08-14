import torch
in_channels,out_channels=5,9
width,height=100,100
kernal_size=3
batch_size=1
# 生成一张(b,c,w,h) 
input=torch.randn(batch_size,in_channels,width,height)
# Conv2d就是二维Convolution层
conv_layer=torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernal_size)
output=conv_layer(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight)
print(conv_layer.weight.shape)
