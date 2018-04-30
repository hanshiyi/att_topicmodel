import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class nvdm_encoder(nn.Module):
    def __init__(self, word_size, hidden_size , topic_num):
        super(nvdm_encoder, self).__init__()
        self.mlp1 = nn.Linear(word_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.mean_layer = nn.Linear(hidden_size, topic_num)
        self.logsigma_layer = nn.Linear(hidden_size, topic_num)
        self.logsigma_layer.weight.data.copy_(torch.zeros(hidden_size, topic_num))
        self.logsigma_layer.bias.data.copy_(torch.zeros(topic_num))

    def forward(self, input):
        hidden = self.activation(self.mlp2(self.activation(self.mlp1(input))))
        # hidden = self.activation(self.mlp(input))
        # hidden = torch.cat((hidden,docMP),1)
        mean = self.mean_layer(hidden)
        logsigma = self.logsigma_layer(hidden)
        return mean, logsigma

class nvdm_ctr(nn.Module):
    def __init__(self, word_size, emb_siz, topic_num, using_gpu):
        super(nvdm_ctr, self).__init__()
        self.word_size = word_size
        self.gid = using_gpu
        self.emb = nn.Embedding(word_size, topic_num)
        self.bias = nn.Parameter(torch.zeros(word_size))

    def forward(self,input):
        emblist = Variable(torch.LongTensor([i for i in range(self.word_size)])).cuda(self.gid)
        emb = torch.transpose(self.emb(emblist), 0, 1)
        output = torch.mm(input, emb)
        output = output + self.bias.expand_as(output)
        return output

    def dataloader(self, filename):
        self.w2idx = {}
        self.w2vfile = open(filename, 'r')
        w2v = []
        idx = 0
        dim = 0
        for line in self.w2vfile.readlines():
            # if dim == 0 and idx != 0:
            #     dim = int(len(line.strip().split()[1:]))
            #     continue
            if len(line.strip().split()) == 2:
                continue
            content = line.strip().split()
            self.w2idx[content[0]] = idx
            idx += 1
            w2v.append([np.float64(i) for i in content[1:]])
        npmatrix = np.array(w2v)
        self.emb.weight.data.copy_(torch.from_numpy(npmatrix))
        return idx, dim

    def predict(self):
        wordlist = Variable(torch.LongTensor([i for i in xrange(0, self.word_size)]))
        emb = self.emb(wordlist)
        return emb




class transitionmodel(nn.Module):
    def __init__(self, word_size, emb_siz, topic_num, filename, using_gpu):
        super(transitionmodel, self).__init__()
        hidden_size=500
        self.word_size = word_size
        self.topic_num = topic_num
        self.emb = nn.Embedding(word_size+1, topic_num)
        self.activation = torch.tanh
        self.mlp = nn.Linear(word_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, topic_num)
        self.logsigma_layer = nn.Linear(hidden_size, topic_num)
        self.logsigma_layer.weight.data.copy_(torch.zeros(hidden_size, topic_num))
        self.logsigma_layer.bias.data.copy_(torch.zeros(topic_num))
        self.bias = nn.Parameter(torch.zeros(word_size))
        self.gid = using_gpu
        self.w2idx = {}
        self.softmax = torch.nn.Softmax()
        self.textencoder = nn.GRU(input_size=topic_num, hidden_size=topic_num)
        with open(filename) as file:
            idxx = 1
            for line in file.readlines():
                self.w2idx[line.strip().split()[0]] = idxx
                idxx += 1

    def word2id(self, winlist):
        ans = []
        for win in winlist:
            ans.append([self.w2idx[i] for i in win])
        return ans

    def forward(self, winlst, bowlist):
        temlist = []
        emblist = Variable(torch.LongTensor([i for i in range(self.word_size)])).cuda(self.gid)
        emb = torch.transpose(self.emb(emblist), 0, 1)
        emb_np = self.emb.weight.data.cpu().numpy()
        docuWE = []
        seq_info = winlst[0]
        seq_len = winlst[1]
        for idx, docu in enumerate(seq_info):
            temto = []
            for i in range(seq_len[idx]):
                emb_wordwindow = emb_np[docu[i]]
                temto.append(np.argmax(emb_wordwindow))
            temlist.append(self.one_hot(temto))
        docuWE = self.emb(Variable(torch.LongTensor(seq_info).cuda(self.gid)))
        packed_docuWE = torch.nn.utils.rnn.pack_padded_sequence(docuWE, seq_len, batch_first=True)
        rnn_h = Variable(torch.zeros(1, int(docuWE.size()[0]), self.topic_num)).cuda(self.gid)
        topic_trans, transH = self.textencoder(packed_docuWE, rnn_h)
        rnn_hidden = torch.squeeze(transH, 0)
        hidden = self.activation(self.mlp(bowlist))
        mean = self.mean_layer(hidden)
        logsigma = self.logsigma_layer(hidden)
        kld = torch.sum(-0.5 * (1 - mean * mean + 2 * logsigma - torch.exp(2 * logsigma)),
                        1)
        EPS = Variable(torch.normal(torch.zeros(len(seq_info), self.topic_num),
                                    torch.ones(len(seq_info), self.topic_num))).cuda(self.gid)
        goal = mean + torch.exp(logsigma) * EPS
        zout = torch.nn.utils.rnn.pad_packed_sequence(topic_trans, batch_first=True)
        topic_attention = self.softmax(Variable(torch.FloatTensor(temlist)).cuda(self.gid))
        output = torch.mm(goal+rnn_hidden * topic_attention, emb)
        output = output + self.bias.expand_as(output)
        return output, kld

    def ppforward(self, input):
        emblist = Variable(torch.LongTensor([i for i in range(self.word_size)])).cuda(0)
        emb = torch.transpose(self.emb(emblist), 0, 1)
        output = torch.mm(input, emb)
        output = output + self.bias.expand_as(output)
        return output

    def one_hot(self, labels):
        t = np.zeros(self.topic_num)
        for i in labels:
            t[i] += 1
        return t

    def dataloader(self, filename):
        self.w2idx = {}
        self.w2vfile = open(filename, 'r')
        w2v = []
        idx = 0
        dim = 0
        for line in self.w2vfile.readlines():
            # if dim == 0 and idx != 0:
            #     dim = int(len(line.strip().split()[1:]))
            #     continue
            if len(line.strip().split()) == 2:
                continue
            content = line.strip().split()
            self.w2idx[content[0]] = idx
            idx += 1
            w2v.append([np.float64(i) for i in content[1:]])
        npmatrix = np.array(w2v)
        self.emb.weight.data.copy_(torch.from_numpy(npmatrix))
        return idx, dim

    def predict(self):
        wordlist = Variable(torch.LongTensor([i for i in xrange(0, self.word_size)]))#.cuda(0)
        emb = self.emb(wordlist)
        return emb



