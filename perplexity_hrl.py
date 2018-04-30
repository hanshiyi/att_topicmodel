import torch
from torch.autograd import Variable
import model.utils as postopic
import numpy as np
import model.hdrl4tm as hdrl4tm

def main():
    iter = 0
    MaxIteration = 1000
    topic_num = 50
    word_num = 2075
    emb_dim = 300
    epsilon = 0.8
    topic_dim = 300
    using_gpu = 1
    batch_size = 64
    learning_rate = 5e-5
    window_size = 1
    hidden_size = 500
    meta_c = hdrl4tm.meta_controler(word_num, hidden_size, topic_num).cuda(using_gpu)
    ctr = hdrl4tm.controler2(word_num, emb_dim, topic_num, 'reuters_word_pure.txt', using_gpu).cuda(using_gpu)
    meta_c.load_state_dict(torch.load('/home/davidwang/rlfortopicmodel/model/reutrans121/meta1998'))
    ctr.load_state_dict(torch.load('/home/davidwang/rlfortopicmodel/model/reutrans121/ctr1998'))
    def to_np(x):
        return x.data.cpu().numpy()
    destr = "reuterstestlist.txt"
    bag_url = "reuterstestbag.txt"
    filelist, randomfilelist = postopic.FileListfromFile(destr)
    batchfilelist = postopic.BatchListFile(randomfilelist, batch_size)
    baglist = postopic.ListBag(bag_url)
    # batches = batches[-2:]
    total_kld = 0.
    total_e_dis = 0.
    sig = torch.nn.LogSoftmax()
    doc = 0
    for b_idx, batch in enumerate(batchfilelist):
        doc += len(batch)
        winlst, batchwordlist = postopic.BatchPickWindowFromFile(batch, baglist, filelist, word_num)
        goal_mean, goal_logsig = meta_c(Variable(torch.FloatTensor(batchwordlist)).cuda(using_gpu))
        test = goal_mean * goal_mean
        kld = torch.sum(-0.5 * (1 - goal_mean * goal_mean + 2 * goal_logsig - torch.exp(2 * goal_logsig)),
                        1)
        EPS = Variable(torch.normal(torch.zeros(len(batchwordlist), topic_num),
                                    torch.ones(len(batchwordlist), topic_num))).cuda(using_gpu)
        goal = goal_mean + torch.exp(goal_logsig) * EPS
        topic_dis = ctr(goal, winlst)
        batchwordlist = np.array(batchwordlist)
        batchwordlist = Variable(torch.FloatTensor(batchwordlist)).cuda(using_gpu)
        word_count = torch.sum(batchwordlist, 1)
        recon_loss = -torch.sum(batchwordlist * sig(topic_dis), 1)
        total_e_dis += torch.sum((recon_loss + kld) / word_count)
        print (np.exp(to_np(total_e_dis) / float(doc)))


if __name__ == '__main__':
    main()