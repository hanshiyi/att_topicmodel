import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import utils.w2vloader as w2vld
import utils.negtopic as negtopic
import utils.leftModule as postopic
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score
from utils.logger import Logger

def main():
    iter = 0
    MaxIteration = 100000
    topic_num = 20
    neg_num = 5
    pos_num = 0
    epsilon = 0.8
    topic_dim = 30
    winsiz = 3
    using_gpu = 1
    batch_size = 5
    learning_rate = 0.0025
    # cuda_valid = torch.cuda.is_available()
    cuda_valid = False
    # w2v_model = w2vld.w2vloader('data/glove.42B.300d.txt', cuda_valid, using_gpu)
    w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    corpus_path = 'data/20news-bydate-train-processed'
    pospart = postopic.multiPercepton(w2v_model.dim, 100, topic_num, winsiz).cuda(using_gpu) if cuda_valid else postopic.multiPercepton(w2v_model.dim, 100, topic_num, winsiz)
    optimizer_pos = optim.SGD(pospart.parameters(), lr=learning_rate)
    criterian = torch.nn.CrossEntropyLoss()
    logger = Logger('./logger/20newmlp')
    avg_accuracy = 0.
    def to_np(x):
        return x.data.cpu().numpy()
    try:
        for iter in range(MaxIteration):
            winsiz_list = [winsiz for _ in xrange(batch_size)]
            cur_win, nxt_win, y_topic = postopic.BatchPick(batch_size, corpus_path, winsiz_list)
            pos_context_emb = w2v_model.load_wordembed(cur_win)
            pos_topic_dis = pospart(pos_context_emb)
            pred_topic = torch.max(pos_topic_dis, 1)[1]
            pos_temp_list = []
            y_topic = np.squeeze(y_topic,1)
            loss = criterian(pos_topic_dis, Variable(torch.LongTensor(y_topic)))
            optimizer_pos.zero_grad()
            loss.backward()
            optimizer_pos.step()
            accuracy = (torch.FloatTensor(y_topic) == torch.squeeze(torch.FloatTensor(to_np(pred_topic)))).float().mean()
            avg_accuracy += accuracy
            if iter % 100 == 0:
                print loss
                avg_accuracy = avg_accuracy/100.0
                print avg_accuracy

                info = {
                    'loss': loss.data[0],
                    'accuracy': avg_accuracy
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, iter + 1)

    except KeyboardInterrupt:
        print 'termination by ' + str(iter) + 'iterations'
        torch.save(pospart.state_dict(), 'model/20news/mlp' + str(iter))

if __name__ == '__main__':
    main()