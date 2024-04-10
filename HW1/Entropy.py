import os
import jieba
#import chardet
import math
import os
from collections import Counter

folder_path = 'jyxstxtqj_downcc.com'

def get_files():
    with open(folder_path + '/inf.txt', encoding='gb18030', mode='r') as f:
        names = str(f.read())
        return names

def word_genaration(name):
    #生成字和词
    txt = open(folder_path + os.sep + name + '.txt', "r", encoding="gb18030").read()
    txt = txt.replace('\n','')
    txt = txt.replace('\u3000', '')
    txt = txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
    with open("cn_stopwords.txt", "r", encoding='utf-8') as f:
        stop_word = f.read().split('\n')
        f.close()
    with open("cn_punctuation.txt", "r", encoding='utf-8') as f:
        punctuation = f.read().split('\n')
        f.close()
    for words in jieba.lcut(txt):
        if (words not in stop_word) and (words not in punctuation):
            split_word.append(words)
    for word in txt:
        if (word not in stop_word) and (word not in punctuation):
            single_word.append(word)
    return single_word,split_word

def get_unigram_tf(words):
    unigram_tf = {}
    for w in words:
        unigram_tf[w] = unigram_tf.get(w, 0) + 1
    return unigram_tf

def get_bigram_tf(words):
    bigram_tf = {}
    for i in range(len(words) - 1):
        bigram_tf[(words[i], words[i + 1])] = bigram_tf.get((words[i], words[i + 1]), 0) + 1
    return bigram_tf

def get_trigram_tf(words):
    trigram_tf = {}
    for i in range(len(words) - 2):
        trigram_tf[(words[i], words[i + 1], words[i + 2])] = trigram_tf.get(
            (words[i], words[i + 1], words[i + 2]), 0) + 1
    return trigram_tf

def char_calculate1(word,is_ci,name):
    word_tf = get_unigram_tf(word)
    word_len = sum([item[1] for item in word_tf.items()])
    entropy = sum(
        [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
         word_tf.items()])
    if is_ci:
        print("<{}>基于词的一元模型的中文信息熵为：{}比特/词".format(name, entropy))
    else:
        print("<{}>基于字的一元模型的中文信息熵为：{}比特/字".format(name, entropy))
    return entropy

def char_calculate2(word,is_ci,name):
    word_tf = get_bigram_tf(word)
    last_word_tf = get_unigram_tf(word)
    bigram_len = sum([item[1] for item in word_tf.items()])#依次对每个二元词出现的次数求和，得到总长度/总数量
    unigram_len=sum([item[1] for item in last_word_tf.items()])
    entropy = []
    for bigram in word_tf.items():
        p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
        #p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
        p_x_y = p_xy / (last_word_tf[bigram[0][0]]/unigram_len) # 条件概率p(x|y)
        entropy.append(-p_xy * math.log(p_x_y, 2))
        #print(bigram[0][0])
    entropy = sum(entropy)
    if is_ci:
        print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(name, entropy))
    else:
        print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(name, entropy))
    return entropy

def char_calculate3(word,is_ci,name):
    # 计算三元模型的信息熵
    # 计算三元模型总词频
    word_tf = get_trigram_tf(word)
    last_word_tf = get_bigram_tf(word)
    trigram_len = sum([item[1] for item in word_tf.items()])
    bigram_len = sum([item[1] for item in last_word_tf.items()])
    entropy = []
    for trigram in word_tf.items():
        p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
        #p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
        p_x_y = p_xy / (last_word_tf[(trigram[0][0], trigram[0][1])]/bigram_len)  # 条件概率p(x|y)
        entropy.append(-p_xy * math.log(p_x_y, 2))
    entropy = sum(entropy)
    if is_ci:
        print("<{}>基于词的三元模型的中文信息熵为：{}比特/词".format(name, entropy))
    else:
        print("<{}>基于字的三元模型的中文信息熵为：{}比特/字".format(name, entropy))
    return entropy

if __name__ == '__main__':
    names=get_files()
    for name in names.split(','):
        split_word = []
        single_word = []
        single_word,split_word=word_genaration(name)
        char_calculate1(single_word, 0, name)
        char_calculate1(split_word, 1, name)
        char_calculate2(single_word, 0, name)
        char_calculate2(split_word, 1, name)
        char_calculate3(single_word, 0, name)
        char_calculate3(split_word, 1, name)