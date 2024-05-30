# coding:utf-8
import json
import os
import chardet
from collections import Counter
from random import randint

import gensim
import jieba
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

folder_path = 'jyxstxtqj_downcc.com'
stopwords_files = ['cn_stopwords.txt','cn_punctuation.txt']

para_num = 250
per_para_words = 600
corpus_context_dict = {}
id_corpus_dict = {}
stopwords_list=[]
#topic_num = 100
GEN_DATA = True

def get_files():
    with open(folder_path + '/inf.txt', encoding='gb18030', mode='r') as f:
        names = str(f.read())
        print(names)
        name_list = [folder_path + os.sep + name + '.txt' for name in names.split(',')]
        return name_list


def get_texts():
    #import_stopwords()
    corpus_context_dict = {}
    id_corpus_dict = {}
    id = 0
    for file in get_files():
        simple_name = str(file).split(os.sep)[1].split('.')[0]
        with open(file, 'rb') as f:
            context = f.read()
            real_encode = chardet.detect(context)['encoding']
            context = context.decode(real_encode, errors='ignore')
            new_context = ''
            for c in context:
                if is_chinese(c):
                    new_context += c
            # for sw in stopwords_list:
            #     new_context = new_context.replace(sw, '')
            ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
                  '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
                  '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....',
                  '......',
                  '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
            for a in ad:
                new_context = new_context.replace(a, '')
            corpus_context_dict[simple_name] = new_context
            id_corpus_dict[id] = simple_name
            id += 1
        print(id)
    return corpus_context_dict, id_corpus_dict

def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

def import_stopwords():
    for sf in stopwords_files:
        with open(sf, encoding='utf-8',mode='r') as f:
            stopwords_list.extend([word.strip('\n') for word in f.readlines()])
    #print(stopwords_list)
    return stopwords_list

def get_dataset(stopwords_list):
    data = []
    for i in range(para_num):
        context = corpus_context_dict[id_corpus_dict[i % 16]]
        rand_value = randint(0, len(context) - per_para_words)
        new_context = context[rand_value:rand_value + per_para_words]
        cut_list = list(jieba.cut(new_context, cut_all=False))
        filter_list = []
        for c in cut_list:
            if stopwords_list.__contains__(c):
                continue
            filter_list.append(c)
        data.append((i % 16, filter_list))
    return data


def cluster(model):
    all_words = []
    for file in os.listdir('train_data'):
        with open('train_data/' + file, encoding='utf-8',mode='r') as f:
            all_words.extend(f.read().split(" "))
    word_times_dict = Counter(all_words)

    highest_words = []
    #print(word_times_dict.items())
    for k, v in word_times_dict.items():
        if v > 80:
            highest_words.append(k)
    word_vectors = []
    for tmp_word in highest_words:
        word_vectors.append(model.wv[tmp_word])
    tSNE = TSNE()
    word_embeddings = tSNE.fit_transform(word_vectors)
    classifier = KMeans(n_clusters=16)
    classifier.fit(word_embeddings)
    labels = classifier.labels_

    min_left = min(word_embeddings[:, 0])
    max_right = max(word_embeddings[:, 0])
    min_bottom = min(word_embeddings[:, 1])
    max_top = max(word_embeddings[:, 1])

    markers = ["bo", "go", "ro", "co", "mo", "yo", "ko", "bx", "gx", "rx", "cx", "mx", "yx", "kx", "b>", "g>"]

    for i in range(len(word_embeddings)):
        plt.plot(word_embeddings[i][0], word_embeddings[i][1], markers[labels[i]])
        #plt.annotate('{}'.format(highest_words[i]), xy=(word_embeddings[i][0], word_embeddings[i][1]),
        #             xytext=(word_embeddings[i][0] + 0.1, word_embeddings[i][1] + 0.1))
    plt.axis([min_left, max_right, min_bottom, max_top])
    plt.savefig("./kmeans_result.png")

if __name__ == '__main__':
    if GEN_DATA:
        corpus_context_dict, id_corpus_dict = get_texts()
        stopwords_list=import_stopwords()
        dataset = get_dataset(stopwords_list)
        for name in corpus_context_dict.keys():
            words = jieba.cut(corpus_context_dict[name], cut_all=False)
            with open('train_data/' + name, 'w', encoding='utf-8') as f:
                for w in words:
                    f.write(w)
                    f.write(" ")
    else:
        #corpus_context_dict, id_corpus_dict = get_texts()
        #word2vec_model_cb = Word2Vec(sentences=PathLineSentences('train_data'), hs=1, min_count=10, window=5, vector_size=200, sg=0, workers=16, epochs=10)
        #word2vec_model_sg = Word2Vec(sentences=PathLineSentences('train_data'), hs=1, min_count=10, window=5, vector_size=200, sg=1, workers=16, epochs=10)
        #word2vec_model_cb.save('cbow.model')
        #word2vec_model_sg.save('skip_gram.model')
        #word2vec_model = Word2Vec.load('skip_gram.model')
        word2vec_model = Word2Vec.load('cbow.model')
        #cluster(word2vec_model)
        # test_name = ['郭靖', '萧峰', '桃花岛','蛤蟆功']
        # for name in test_name:
        #     print(name)
        #     for result in word2vec_model.wv.similar_by_word(name, topn=10):
        #         print(result[0], '{:.3f}'.format(result[1]))

        test_paragraph1="郭杨二人见他背上负着一个包裹，甚是累赘，斗了一会，一名武官钢刀砍去，削在他包裹之上，当啷一声，包裹破裂，散出无数物事。曲三乘他欢喜大叫之际，右拐挥出，啪的一声，一名武官顶门中拐，扑地倒了。余下那人大骇，转身便逃。他脚步甚快，顷刻间奔出数丈。曲三右手往怀中一掏，跟着扬手，月光下只见一块圆盘似的黑物飞将出去，托的一下轻响，嵌入了那武官后脑。那武官惨声长叫，单刀脱手飞出，双手乱舞，仰天缓缓倒下，扭转了几下，就此不动，眼见是不活了。"
        test_paragraph2="郭和杨两个人看到他肩上背着一个包裹，十分累赘，和武官过招了一会，其中一名武官使用钢刀向他砍去，当啷一声，砍在了包裹上，包裹破裂，掉落出很多东西。曲三乘他欣喜大叫的时候，挥出右拐，一名武官的顶门中拐，扑地倒了。剩下那个人大惊失色，转身就逃。曲三右手往怀中一掏，顺手一扬，只见一块圆盘似的黑物飞出去，托的一下轻响，嵌入了逃跑的那位武官后脑。那位武官惨声长叫，仰天缓缓倒下，痉挛了几下，就此不动，眼见是死得彻底了。"
        # 计算段落相似度
        similarity_score=word2vec_model.wv.wmdistance(test_paragraph1, test_paragraph2)
        #wmdistance比较语句之间的相似度，数值越大代表越不相似
        print(f"段落1与段落2的语义相似度：{similarity_score}")
