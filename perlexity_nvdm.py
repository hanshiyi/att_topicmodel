import torch
import torch.nn as nn
from torch.autograd import Variable
import model.w2vloader as w2vld
import model.utils as postopic
import random
import time
import numpy as np
from torch import optim
import model.hdrl4tm as hdrl4tm
import readdata

def main():
    iter = 0
    MaxIteration = 1000
    topic_num = 50
    word_num = 2075
    emb_dim = 300
    epsilon = 0.8
    topic_dim = 300
    using_gpu = 0
    batch_size = 64
    learning_rate = 5e-5
    hidden_size = 500
    meta_c = hdrl4tm.nvdm_encoder(word_num, hidden_size, topic_num)
    ctr = hdrl4tm.nvdm_ctr(word_num, emb_dim, topic_num, using_gpu)
    # ctr = hdrl4tm.nvdm_ctr(word_num, emb_dim, topic_num, '/home/shiyi/nvdm/data/20news/vocab.new', using_gpu)
    meta_c.load_state_dict(torch.load('/home/davidwang//rlfortopicmodel/model/reuterssnvdm103/meta999'))
    ctr.load_state_dict(torch.load('/home/davidwang//rlfortopicmodel/model/reuterssnvdm103/ctr999'))
    def to_np(x):
        return x.data.cpu().numpy()
    data_url = "reuterstestbag.txt"
    batches = readdata.CreateBatches(data_url, batch_size, shuffle=True)
    # batches = batches[-2:]
    total_kld = 0.
    total_e_dis = 0.
    sfm = torch.nn.LogSoftmax()
    sig = torch.nn.Sigmoid()
    doc = 0
    for b_idx, batch in enumerate(batches):
        doc += len(batch)
        batchwordlist, batchwordoccur = readdata.PickInBatch(data_url, batch, batch_size, word_num, mode=1) #64*10000/2000 64*words
        goal_mean, goal_logsig = meta_c(Variable(torch.FloatTensor(batchwordlist)))
        kld = torch.sum(-0.5*(1 - goal_mean*goal_mean + 2 * goal_logsig - torch.exp(2 * goal_logsig)), 1)
        torch.manual_seed(int(time.time()))
        EPS = Variable(
            torch.normal(torch.zeros(len(batchwordlist), topic_num), torch.ones(len(batchwordlist), topic_num)))
        goal = goal_mean + torch.exp(goal_logsig) * EPS
        topic_dis = ctr(goal)
        batchwordlist = np.array(batchwordlist)
        batchwordlist = Variable(torch.FloatTensor(batchwordlist))
        word_count = torch.sum(batchwordlist,1)
        recon_loss = -torch.sum(batchwordlist * sfm(topic_dis), 1)
        total_e_dis += torch.sum((recon_loss+kld)/word_count)
        print (np.exp(to_np(total_e_dis)/float(doc)))


if __name__ == '__main__':
    main()