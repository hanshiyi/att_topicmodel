import torch.nn as nn
import torch.autograd  as autograd
import torch
import numpy as np

class w2vloader(nn.Module):
    def __init__(self,filename, cuda_valid, using_gpu):
        super(w2vloader, self).__init__()
        self.w2vfile = open(filename, 'r')
        self.using_gpu = using_gpu
        self.cuda_valid = cuda_valid
        num, dim, npmatrix = self.__dataloader()
        self.dim = dim
        self.emb = nn.Embedding(num, dim).cuda(using_gpu) if cuda_valid else nn.Embedding(num, dim)
        self.emb.weight.data.copy_(torch.from_numpy(npmatrix))

    def __dataloader(self):
        self.w2idx = {}
        w2v = []
        idx = 0
        dim = 0
        for line in self.w2vfile.readlines():
            if dim == 0 and idx!=0:
                dim = int(len(line.strip().split()[1:]))
                continue
            if len(line.strip().split()) == 2:
                continue
            content = line.strip().split()
            self.w2idx[content[0]] = idx
            idx += 1
            w2v.append([np.float64(i) for i in content[1:]])
        return idx, dim, np.array(w2v)

    def load_wordembed(self,wordlist):
        if isinstance(wordlist, str):
            return self.emb(autograd.Variable(torch.LongTensor([self.w2idx[wordlist]])))
        temp = np.array(wordlist)
        if temp.ndim == 1:
            extraced_list = []
            for word in wordlist:
                if word in self.w2idx:
                    extraced_list.append(self.w2idx[word])
            idxlist = autograd.Variable(torch.LongTensor(extraced_list)).cuda(
                self.using_gpu) if self.cuda_valid else autograd.Variable(torch.LongTensor(extraced_list))
            return self.emb(idxlist)
        if temp.ndim==2:
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
            idxlist = autograd.Variable(torch.LongTensor(extraced_list)).cuda(self.using_gpu) if self.cuda_valid else autograd.Variable(torch.LongTensor(extraced_list))
            return self.emb(idxlist)
        if temp.ndim==3:
            wordlist = temp.transpose((1,0,2))
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
                temp_idx_list = autograd.Variable(torch.LongTensor(lll)).cuda(
                    self.using_gpu) if self.cuda_valid else autograd.Variable(torch.LongTensor(lll))
                temp_idx_list = self.emb(temp_idx_list)
                idx_list.append(temp_idx_list)
            return idx_list

    def normalize(self,input):
        return input / input.norm(2,-1).clamp(min=3).expand_as(input)
def to_np(x):
    return x.data.cpu().numpy()

if __name__ == '__main__':
    test_model = w2vloader('../data/lilittleglovetraintest', False,0)
    # print test_model.load_wordembed('happy')/home/shiyi/nvdm/data/20news/vocab.new
    file = open('/home/shiyi/nvdm/data/20news/vocab.new','r')
    linelist = [line.strip().split()[0] for line in file.readlines() if line.strip().split()[0] in test_model.w2idx]
    outfile = open('../data/wholevocab2000', 'w')
    for word in linelist:
        outfile.write(word+'\n')
    outfile = open('../data/top200020newsglove','w')
    for line in linelist:
        if line.strip() in test_model.w2idx:
            outfile.write(line.strip()+' ' + ' '.join([str(i) for i in to_np(test_model.load_wordembed(line.strip()))[0]])+'\n')