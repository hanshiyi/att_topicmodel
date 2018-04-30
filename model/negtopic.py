import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable


class negtopic(nn.Module):
    def __init__(self, dim, topic_num, window):
        super(negtopic, self).__init__()
        self.window_size = window
        self.rnn = nn.GRU(dim, dim / 2, 1, bidirectional=True, batch_first=True)
        self.fn = nn.Linear(dim, topic_num, bias=False)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden, cen_idx):
        output, h_next = self.rnn(input, hidden)
        max_pool = output.mean(1)
        # center_word = torch.index_select(input, 1, cen_idx)
        # cat_vec = torch.cat((torch.squeeze(center_word, 1), torch.squeeze(max_pool, 1)), 1)
        topic_prob = self.softmax(self.fn(torch.squeeze(max_pool, 1)))
        return topic_prob

    def predict(self, input):
        input = torch.cat((input, Variable(torch.zeros(input.size()))),1)
        return self.softmax(self.fn(input))

    def copy_from(self, brother):
        self.rnn.load_state_dict(brother.rnn.state_dict())
        self.fn.weight.data.copy_(brother.fn.weight.data)
        return


class negtopic_nocat(nn.Module):
    def __init__(self, dim, topic_num, window):
        super(negtopic_nocat, self).__init__()
        self.window_size = window
        self.rnn = nn.GRU(dim, dim / 2, 1, bidirectional=True, batch_first=True)
        self.fn = nn.Linear(dim, topic_num, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden, cen_idx):
        output, h_next = self.rnn(input, hidden)
        max_pool = torch.max(output, 1)[0]
        topic_prob = self.softmax(self.fn(torch.squeeze(max_pool, 1)))
        return topic_prob

    def copy_from(self, brother):
        self.rnn.load_state_dict(brother.rnn.state_dict())
        self.fn.weight.data.copy_(brother.fn.weight.data)
        return


class MUSE(nn.Module):
    def __init__(self, dim, topic_num, window):
        super(MUSE, self).__init__()
        self.window_size = window
        self.fn = nn.Linear(dim+dim, topic_num)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden, cen_idx):
        center_word = torch.index_select(input, 1, cen_idx)
        input = input.mean(1)
        cat_vec = torch.cat((torch.squeeze(center_word, 1), torch.squeeze(input, 1)), 1)
        topic_prob = self.softmax(self.fn(torch.squeeze(cat_vec, 1)))
        return topic_prob

    def copy_from(self, brother):
        self.fn.weight.data.copy_(brother.fn.weight.data)
        return

class MUSE_sum(nn.Module):
    def __init__(self, dim, topic_num, word_num):
        super(MUSE_sum, self).__init__()
        self.fn = nn.Linear(dim, topic_num, bias=False)
        self.fn = nn.Parameter(torch.randn(dim, topic_num).normal_(0, 1.5), requires_grad=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        self.emb = nn.Embedding(word_num, dim)

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

    def load_wordembed(self, wordlist):
        if isinstance(wordlist, str):
            return self.emb(autograd.Variable(torch.LongTensor([self.w2idx[wordlist]])))
        temp = np.array(wordlist)
        if temp.ndim == 1:
            extraced_list = []
            for word in wordlist:
                if word in self.w2idx:
                    extraced_list.append(self.w2idx[word])
            idxlist = autograd.Variable(torch.LongTensor(extraced_list)).cuda(3)
                # self.using_gpu) if self.cuda_valid else autograd.Variable(torch.LongTensor(extraced_list))
            return self.emb(idxlist)
        if temp.ndim == 2:
            extraced_list = []
            batch_size = len(wordlist)
            for wl in wordlist:
                idx_list = []
                for i in wl:
                    if i not in self.w2idx:
                        idx_list.append(0)
                    else:
                        idx_list.append(self.w2idx[i])
                extraced_list.append(idx_list)
            idxlist = autograd.Variable(torch.LongTensor(extraced_list)).cuda(3)
                # self.using_gpu) if self.cuda_valid else autograd.Variable(torch.LongTensor(extraced_list))
            return self.emb(idxlist)
        if temp.ndim == 3:
            wordlist = temp.transpose((1, 0, 2))
            extraced_list = []
            for ba_wl in wordlist:
                idx_ba = []
                for bag in ba_wl:
                    idx_win = []
                    for window_word in bag:
                        idx_win.append(self.w2idx[window_word])
                    idx_ba.append(idx_win)
                extraced_list.append(idx_ba)
            idx_list = []
            for lll in extraced_list:
                temp_idx_list = autograd.Variable(torch.LongTensor(lll)).cuda(3)
                    # self.using_gpu) if self.cuda_valid else autograd.Variable(torch.LongTensor(lll))
                temp_idx_list = self.emb(temp_idx_list)
                idx_list.append(temp_idx_list)
            return idx_list

    def forward(self, input, hidden, cen_idx):
        input = input.mean(1)
        # diff = torch.squeeze(input, 1)-self.fn
        # e_dis = torch.cumsum(diff*diff,1)
        dot_product = torch.mm(torch.squeeze(input, 1),self.fn)
        topic_prob = self.sigmoid(dot_product)
        soft_topic_prob = self.softmax(dot_product)
        # self.fn.data.copy_(self.normalize(self.fn.data))
        return topic_prob,soft_topic_prob

    def normalize(self):
        self.fn.data.copy_(self.fn.data / self.fn.data.norm(2,-1).clamp(min=3).expand_as(self.fn.data))

    def predict(self, input):
        # dot_product = self.fn(input, 1)
        dot_product = input.mm(self.fn)
        topic_prob = self.sigmoid(dot_product)
        return topic_prob

    def copy_from(self, brother):
        self.fn.data.copy_(brother.fn.data)
        return


if __name__ == '__main__':
    testnet_1 = negtopic(300)
    testnet_2 = negtopic(300)
    import w2vloader
    test_model = w2vloader.w2vloader('../data/w2v20news.txt')
    hidden = torch.zeros(test_model.load_wordembed('ball').size())
    print testnet_1(test_model.load_wordembed('ball'), hidden)
    print testnet_2(test_model.load_wordembed('ball'), hidden)
    testnet_1.copy_from(testnet_2)
    print testnet_1(test_model.load_wordembed('ball'), hidden)