import gensim
import sys
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

datapath = sys.argv[1]
file = open(datapath,'r')
model = Word2Vec(LineSentence(file), size=300, window=5, min_count=1, workers=multiprocessing.cpu_count()-8, iter=20)
model.wv.save_word2vec_format('w2v20news.txt', binary=False)