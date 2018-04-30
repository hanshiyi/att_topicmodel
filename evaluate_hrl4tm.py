import torch
import numpy as np
import model.hdrl4tm as hdrl4tm
import heapq

def main():
    topic_num = 50
    word_num = 2075
    emb_dim = 300
    hidden_size = 500
    using_gpu = 3
    meta_c = hdrl4tm.meta_controler(word_num, hidden_size, topic_num)
    ctr = hdrl4tm.controler2(word_num, emb_dim, topic_num, 'reuters_word_pure.txt', using_gpu)
    meta_c.load_state_dict(torch.load('/home/davidwang/rlfortopicmodel/model/reutrans121/meta1998'))
    ctr.load_state_dict(torch.load('/home/davidwang/rlfortopicmodel/model/reutrans121/ctr1998'))
    # wordfile = open('/home/shiyi/rlfortopicmodel/data/wholeVocabList+freq_top10000_813.txt', 'r')
    wordfile = open('reuters_word_pure.txt', 'r')
    wordlist = [line.strip().split()[0] for line in wordfile.readlines()]
    outfile = open('data/wordmatrixhrl1020.txt', 'w')
    def to_np(x):
        return x.data.cpu().numpy()
    topic_dis = ctr.predict()
    dis_nparray = np.transpose(to_np(topic_dis), (1, 0))
    for idx, row in enumerate(dis_nparray):
        top_word = heapq.nlargest(50, range(len(row)), row.take)
        print str(idx) + ':',
        for word_id in top_word:
            print wordlist[word_id],
        print ''

    for idx, line in enumerate(wordlist):
        outfile.write(line)
        for i in topic_dis[idx]:
            outfile.write(' '+str(i.data.cpu().numpy().tolist()[0]))
        outfile.write('\n')

if __name__ == '__main__':
    main()
