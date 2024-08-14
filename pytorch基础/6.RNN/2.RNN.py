#使用RNN
import torch
 
input_size=4
hidden_size=4
num_layers=1
batch_size=1
seq_len=5
# 准备数据
idx2char=['e','h','l','o']
x_data=[1,0,2,2,3] # hello
y_data=[3,1,2,3,2] # ohlol
 
one_hot_lookup=[[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]] #分别对应0,1,2,3项
x_one_hot=[one_hot_lookup[x] for x in x_data] # 组成序列张量
print('x_one_hot:',x_one_hot)
 
# 构造输入序列和标签
inputs=torch.Tensor(x_one_hot).view(seq_len,batch_size,input_size)
labels=torch.LongTensor(y_data)  #labels维度是: (seqLen * batch_size ，1)
 
# design model
class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(Model, self).__init__()
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rnn=torch.nn.RNN(hidden_size=self.hidden_size,input_size=self.input_size,
                              num_layers=self.num_layers)
 
    def forward(self,input):
        hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out, _=self.rnn(input,hidden)
        # 这里吧三维out转为二维了，out应该有num_layers,batch_size,hidden_size
        # 为了能和labels做交叉熵，需要reshape一下:(seqlen*batchsize, hidden_size),即二维向量，变成一个矩阵
        # print(out)
        # print(out.view(-1,self.hidden_size))
        return out.view(-1,self.hidden_size)
 
net=Model(input_size,hidden_size,batch_size,num_layers)
 
# loss and optimizer
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(), lr=0.05)
 
# train cycle
for epoch in range(20):
    optimizer.zero_grad()
    #inputs维度是: (seqLen, batch_size, input_size) labels维度是: (seqLen * batch_size * 1)
    #outputs维度是: (seqLen, batch_size, hidden_size)
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    _, idx=outputs.max(dim=1)
    idx=idx.numpy()
    print('Predicted: ',''.join([idx2char[x] for x in idx]),end='')
    print(',Epoch [%d/20] loss=%.3f' % (epoch+1, loss.item()))
 