import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import os
import re
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
MAX_LEN = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenlize(sentence):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)  # 当语料量足够多时，可以学习到I'm的含义。
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result

class ImdbDataset(Dataset):
    def __init__(self, wordSequence=None, train=True):
        super(ImdbDataset,self).__init__()
        self.wordSequence = wordSequence
        data_path = r"./data/aclImdb"
        data_path += r"/train" if train else r"/test" # 文件名拼接【等价于os.path.join()】
        self.total_path = []  # 保存所有的文件路径
        for temp_path in [r"/pos", r"/neg"]:
            cur_path = data_path + temp_path
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]  # 将所有文件路径加入到total_path列表中

    def __getitem__(self, idx):
        file = self.total_path[idx]
        review = tokenlize(open(file, encoding="utf-8").read())  # 读取文件内容（评论）
        label = int(file.split("_")[-1].split(".")[0])
        label = 0 if label < 5 else 1
        if self.wordSequence is not None:
            review = self.wordSequence.transform(review, max_len=MAX_LEN)  #  将字符串通过已经保存的“词语-数字”映射器转为数字
        return review, label

    def __len__(self):
        return len(self.total_path)


def collate_fn(batch):
    reviews, labels = zip(*batch)
    lengths = [len(review) if len(review) < MAX_LEN else MAX_LEN for review in reviews]
    reviews, labels = torch.LongTensor(np.array(list(reviews))),torch.LongTensor(np.array(list(labels)))  # 将tuple类型转为Tensor类型
    return reviews, labels, lengths


def get_dataloader(dataset, train=True):
    batch_size = BATCH_SIZE_TRAIN if train else BATCH_SIZE_TEST
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


class WordSequence:
    PAD_TAG = "<PAD>"  # 句子长度不够时的填充符
    UNK_TAG = "<UNK>"  # 表示未在词典库里出现的未知词汇
    PAD = 0
    UNK = 1

    def __init__(self):
        self.word_index_dict = {self.UNK_TAG:self.UNK, self.PAD_TAG:self.PAD}  # 初始化词语-数字映射字典
        self.index_word_dict = {}  # 初始化数字-词语映射字典
        self.word_count_dict = {}  # 初始化词语-词频统计字典
        self.fited = False

    def __len__(self):
        return len(self.word_index_dict)


    # 接受句子，统计词频得到
    def fit(self,sentence,min_count=5,max_count=None,max_features=None):    # 【min_count:最小词频; max_count: 最大词频; max_features: 最大词语数(词典容量大小)】
        for word in sentence:
            self.word_count_dict[word] = self.word_count_dict.get(word,0)  + 1  #所有的句子fit之后，self.word_count_dict就有了所有词语的词频
        if min_count is not None:   # 根据条件统计词频
            self.word_count_dict = {word:count for word,count in self.word_count_dict.items() if count >= min_count}
        if max_count is not None:#   根据条件统计词频
            self.word_count_dict = {word:count for word,count in self.word_count_dict.items() if count <= max_count}    # 根据条件构造词典
        if max_features is not None:    # 根据条件保留高词频词语
            self.word_count_dict = dict(sorted(self.word_count_dict.items(),key=lambda x:x[-1],reverse=True)[:max_features])    # 保留词频排名靠前的词汇【self.word_count_dict.items()为待排序的对象，key表示排序指标，reverse=True表示降序排列】
        for word in self.word_count_dict:   # 根据word_count_dict字典构造词语-数字映射字典
            if word not in self.word_index_dict.keys(): # 如果当前词语word还没有添加到word_index_dict字典，则添加
                self.word_index_dict[word]  = len(self.word_index_dict)  # 每次word对应一个数字【使用self.word_index_dict添加当前word前已有词汇的数量作为其value】
        self.fited = True
        self.index_word_dict = dict(zip(self.word_index_dict.values(),self.word_index_dict.keys()))  #把word_index_dict进行翻转【准备一个index->word的字典】

    # word -> index
    def to_index(self,word):
        assert self.fited == True,"必须先进行fit操作"
        return self.word_index_dict.get(word,self.UNK)

    # 把句子转化为数字数组(向量)【输入：[str,str,str]；输出：[int,int,int]】
    def transform(self,sentence,max_len=None):
        if len(sentence) > max_len: # 句子过长，截取句子
            sentence = sentence[:max_len]
        else:   # 句子过短，填充句子
            sentence = sentence + [self.PAD_TAG] *(max_len- len(sentence))
        index_sequence = [self.to_index(word) for word in sentence]
        return index_sequence

    # index -> word
    def to_word(self,index):
        assert self.fited , "必须先进行fit操作"
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    # 把数字数组(向量)转化为句子【输入：[int,int,int]；输出：[str,str,str]】
    def inverse_transform(self,indexes):
        sentence = [self.index_word_dict.get(index,"<UNK>") for index in indexes]
        return sentence



def fit_save_word_sequence():
    dataloader_train = get_dataloader(True) # 训练集批次化数据【文本类型】
    dataloader_test = get_dataloader(False) # 测试集批次化数据【文本类型】
    ws = WordSequence()  # 实例化文本序列化对象
    for reviews, label in tqdm(dataloader_train, total=len(dataloader_train)):  # tqdm的作用是提供运行进度条提示
        for review in reviews:
            ws.fit(review)
    for reviews, label in tqdm(dataloader_test, total=len(dataloader_test)):
        for review in reviews:
            ws.fit(review)
    print("构造的词典的容量大小：len(ws) = {0}".format(len(ws)))
    pickle.dump(ws, open("./models/ws.pkl", "wb"))  # 保存文本序列化对象


class LSTMModel(nn.Module):
    def __init__(self, wordSequence, max_len=MAX_LEN):
        super(LSTMModel,self).__init__()
        self.hidden_size = 64
        self.embedding_dim = 200
        self.num_layer = 2
        self.bidriectional = True
        self.bi_num = 2 if self.bidriectional else 1
        self.dropout = 0.5

        self.embedding = nn.Embedding(len(wordSequence),self.embedding_dim,padding_idx=wordSequence.PAD) #[N,300]
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layer,bidirectional=True,dropout=self.dropout)
        #使用两个全连接层，中间使用relu激活函数
        self.fc = nn.Linear(self.hidden_size*self.bi_num,20)
        self.fc2 = nn.Linear(20,2)


    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1,0,2) #进行轴交换
        h_0,c_0 = self.init_hidden_state(x.size(1))
        _,(h_n,c_n) = self.lstm(x,(h_0,c_0))

        #只要最后一个lstm单元处理的结果，这里多去的hidden state
        out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out,dim=-1)

    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        return h_0,c_0



def train(epoch, wordSequence):
    lstm_model.train()
    dataset_train = ImdbDataset(wordSequence=wordSequence, train=True)
    dataloader_train = get_dataloader(dataset=dataset_train, train=True)  # 训练集批次化数据【文本类型】
    for batch_index, (reviews, labels, lengths) in enumerate(dataloader_train):
        reviews = reviews.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = lstm_model(reviews)
        loss = criterion(output, labels)  # traget需要是[0,9]，不能是[1-10]
        loss.backward()
        optimizer.step()
        if batch_index % 130 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index * len(reviews), len(dataloader_train.dataset),100. * batch_index / len(dataloader_train), loss.item()))
            torch.save(lstm_model.state_dict(), "./models/mnist_net{0}.pkl".format(epoch))
            torch.save(optimizer.state_dict(), './models/mnist_optimizer{0}.pkl'.format(epoch))


def test(wordSequence):
    test_loss = 0
    correct = 0
    lstm_model.eval()
    dataset_test = ImdbDataset(wordSequence=wordSequence, train=False)
    dataloader_test = get_dataloader(dataset=dataset_test, train=False)  # 测试集批次化数据【文本类型】
    with torch.no_grad():
        for batch_index, (reviews, labels, lengths) in enumerate(dataloader_test):
            reviews = reviews.to(device)
            labels = labels.to(device)
            output = lstm_model(reviews)
            test_loss += F.nll_loss(output, labels, reduction="sum")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(labels.data).sum()
        test_loss = test_loss / len(dataloader_test.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(dataloader_test.dataset), 100. * correct / len(dataloader_test.dataset)))

if __name__ == '__main__':
    ws = WordSequence()
    ws = pickle.load(open("./models/ws.pkl", "rb"))

    lstm_model = LSTMModel(wordSequence=ws, max_len=MAX_LEN).to(device) #在gpu上运行，提高运行速度
    print(lstm_model)
    optimizer = optim.Adam(lstm_model.parameters())
    criterion = nn.NLLLoss()

    # test()
    for epoch in range(5):
        train(wordSequence=ws, epoch=epoch)
        test(wordSequence=ws)
