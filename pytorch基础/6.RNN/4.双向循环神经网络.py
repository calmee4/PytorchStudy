import torch
import time
import csv
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 GPU 随机种子
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，也设置所有 GPU 的种子e

set_seed(42)  # 选择一个固定的种子值
# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = False  # 修改为 False 以使用 CPU

class NameDataset():  # 处理数据集
    def __init__(self, is_train_set=True):
        filename = 'names_train.csv' if is_train_set else 'names_test.csv'
        with open(filename, 'r') as f:  # 打开压缩文件并将变量名设为为f
            reader = csv.reader(f)  # 读取表格文件
            rows = list(reader)
        self.names = [row[0] for row in rows]  # 取出人名
        self.len = len(self.names)  # 人名数量
        self.countries = [row[1] for row in rows]  # 取出国家名
        self.country_list = list(sorted(set(self.countries)))  # 国家名集合，18个国家名的集合
        self.country_dict = self.getCountryDict()  # 转变成词典
        self.country_num = len(self.country_list)  # 得到国家集合的长度18

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()  # 创建空字典
        for idx, country_name in enumerate(self.country_list, 0):  # 取出序号和对应国家名
            country_dict[country_name] = idx  # 把对应的国家名和序号存入字典
        return country_dict

    def idx2country(self, index):  # 返回索引对应国家名
        return self.country_list(index)

    def getCountrysNum(self):  # 返回国家数量
        return self.country_num

trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountrysNum()  # 模型输出大小

# def create_tensor(tensor):  # 移除GPU判断，直接返回tensor
#     return tensor

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size  # 包括下面的n_layers在GRU模型里使用
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # input.shape=(seqlen,batch) output.shape=(seqlen,batch,hiddensize)
        self.gru = torch.nn.RNN(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)  # 双向GRU会输出两个hidden，维度需要✖2，要接一个线性层

    def forward(self, input, seq_lengths):
        input = input.t()  # input shape :  Batch x Seq -> S x B 用于embedding
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)
        # print(seq_lengths)
        # print(' e ',embedding.size())
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)
        # print(' g ',gru_input.data.size())
        output, hidden = self.gru(gru_input, hidden)  # 双向传播的话hidden有两个
        # print(' o ',output)
        # print(' h ',hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden
def name2list(name):  # 把每个名字按字符都变成ASCII码
    arr = [ord(c) for c in name]
    return arr, len(arr)

def make_tensors(names, countries):  # 处理名字ASCII码 重新排序的长度和国家列表
    sequences_and_lengths = [name2list(name) for name in names]
    # print(sequences_and_lengths)
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return seq_tensor, seq_lengths, countries

def trainModel():
    total_loss = 0

    for i, (names, countries) in enumerate(trainloader, 1):
        optimizer.zero_grad()
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i == len(trainset) // BATCH_SIZE:
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss

def testModel():
    correct = 0
    total = len(testset)

    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total

if __name__ == '__main__':
    print("Train for %d epochs..." % N_EPOCHS)
    start = time.time()
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)

    criterion = torch.nn.CrossEntropyLoss()  # 计算损失
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 更新

    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        print('%d / %d:' % (epoch, N_EPOCHS))
        trainModel()
        acc = testModel()
        acc_list.append(acc)
    end = time.time()
    print(datetime.timedelta(seconds=(end - start) // 1))

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    