#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13:43:15 2015-11-25

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import codecs
import math
import time
from MultiPerceptron import MultiPerceptron as MP
from Dict import Dict
from cPickle import dump
from cPickle import load


CHANGED = "Updated on 13:53 2015-12-17"


class CWSPerceptron:

    def __init__(self):
        self.corpus = list()  # save the corpus for training
        self.tag = list()     # the tag of corpus
        self.corpus_num = 0
        self.state = ['B', 'M', 'E', 'S']
        self.perceptron = MP()
        self.dict = Dict()
        self.init_prb = {'B': 0, 'M': 0, 'E': 0, 'S': 0}
        self.trans_prb = {
            'B': {'B': 0, 'M': 0, 'E': 0, 'S': 0},
            'M': {'B': 0, 'M': 0, 'E': 0, 'S': 0},
            'E': {'B': 0, 'M': 0, 'E': 0, 'S': 0},
            'S': {'B': 0, 'M': 0, 'E': 0, 'S': 0}
        }
        self.dimension = 0
        self.unigram_feat_num = 0
        self.unigram_feat_id = {}
        self.bigram_feat_num = 0
        self.bigram_feat_id = {}
        self.dict_feat_num = 0
        self.dict_feat_id = {}
        self.path = r'./'

    def setSavePath(self, path):
        self.path = path
        self.perceptron.setSavePath(path)

    def saveModel(self):
        print "Saving the unigram&bigram infomation......"
        output1 = open(self.path + r"bigram_feat_id.pkl", 'wb')
        dump(self.bigram_feat_id, output1, -1)
        output1.close()
        output2 = open(self.path + r"unigram_feat_id.pkl", 'wb')
        dump(self.unigram_feat_id, output2, -1)
        output2.close()
        output3 = open(self.path + r"dict_feat_id.pkl", 'wb')
        dump(self.dict_feat_id, output3, -1)
        output3.close()

        # release the memory
        self.unigram_feat_id = []
        self.bigram_feat_id = []
        self.corpus = []
        self.tag = []
        print "Saving the inital prb & trans prb infomation....."
        output1 = open(self.path + r"init_prb.pkl", 'wb')
        dump(self.init_prb, output1, -1)
        output1.close()
        output2 = open(self.path + r"trans_prb.pkl", 'wb')
        dump(self.trans_prb, output2, -1)
        output2.close()
        print "Saving process done."

    def loadModel(self):
        print "Loading the unigram&bigram infomation......"
        inputs = open(self.path + r"bigram_feat_id.pkl", 'rb')
        self.bigram_feat_id = load(inputs)
        self.bigram_feat_num = len(self.bigram_feat_id)
        inputs.close()
        inputs1 = open(self.path + r"unigram_feat_id.pkl", 'rb')
        self.unigram_feat_id = load(inputs1)
        self.unigram_feat_num = len(self.unigram_feat_id)
        inputs1.close()
        inputs2 = open(self.path + r"dict_feat_id.pkl", 'rb')
        self.dict_feat_id = load(inputs2)
        self.dict_feat_num = len(self.dict_feat_id)
        # print "Loading process done."
        print "Loading the prb infomation......"
        inputs = open(self.path + r"init_prb.pkl", 'rb')
        self.init_prb = load(inputs)
        inputs.close()
        inputs1 = open(self.path + r"trans_prb.pkl", 'rb')
        self.trans_prb = load(inputs1)
        inputs1.close()
        print "Loading process done."
        self.dimension = self.unigram_feat_num * 5 + \
            self.bigram_feat_num * 5 + self.dict_feat_num * 4

    def loadDict(self, dictfile):
        self.dict.loadDict(dictfile)

    def saveDict(self, outfile):
        self.dict.saveDict(outfile)

    def readDict(self, dictfile):
        self.dict.readDict(dictfile)

    def appendDict(self, dictfile):
        self.dict.appendDict(dictfile)

    def evaluate(self, corpus=200):
        error_count = 0
        tagnums = sum([len(item) for item in self.tag[0:corpus]])
        for i in range(corpus):
            tag = self.ViterbiDecode(self.corpus[i])
            # print 'y:',self.tag[i]
            # print 'p:',tag
            for index in range(len(tag)):
                pre = tag[index]
                # print self.tag[j]
                real = self.tag[i][index]
                # print pre, real
                if pre != real:
                    error_count += 1
        return 1 - float(error_count) / tagnums

    def segmentation(self, outfile):
        output = codecs.open(outfile, 'w', 'utf-8')
        start = time.clock()
        for i in range(self.corpus_num):
            taglist = self.ViterbiDecode(self.corpus[i])
            wordlist = self.tag2word(self.corpus[i], taglist)
            for j in range(len(wordlist)):
                output.write(wordlist[j])
                output.write(' ')
            output.write("\n")
        print "Decode:", time.clock() - start
        output.close()

    def train(self, trainfile, batch_num=100, max_iter=200, learn_rate=1.0,
              delta_thrd=0.001, is_average=True):
        # self.makelibsvmdata(r'train.data',max_corpus)
        print "Start training process."
        self.perceptron.loadFeatSize(self.dimension, len(self.state))
        self.perceptron.read_train_file(trainfile)
        self.perceptron.printinfo()
        self.perceptron.train_sgd(max_iter, learn_rate, delta_thrd, is_average)
        self.perceptron.saveModel()
        print "Training process done."
        print "Multi-class Perceptron Model had been saved."

    def printstr(self, wordlist):
        for item in wordlist:
            print item
        print " "

    def makeLibSvmData(self, output_file, corpus_num=-1):
        print "Making training data.",
        filecount = 1
        output_data = codecs.open(output_file, 'w')
        if corpus_num == -1:
            corpus_num = self.corpus_num
        for i in range(corpus_num):
            taglist = self.tag[i]
            features = self.GetFeature(self.corpus[i])
            vec = self.Feature2Vec(features)
            for j in range(len(taglist)):
                output_data.write(str(self.state.index(taglist[j])))
                output_data.write('\t')
                keyset = list(vec[j].keys())
                keyset = sorted(keyset)
                if len(keyset) < 1:
                    output_data.write('0:1')
                for key in keyset:
                    output_data.write(str(key))
                    output_data.write(':')
                    output_data.write(str(vec[j][key]))
                    output_data.write(' ')
                output_data.write("\n")
        output_data.close()
        print "\nMaking training data finished."
        return filecount

    def classifiy_score(self, featureVec):
        return self.perceptron.scoreout(featureVec)
        # return self.perceptron.probout(featureVec)

    def getEmitPrb(self, score):
        """
        Get emits_prb use softmax function
        """
        max_score = max(score.values())
        emit_prb = {}
        expsum = 0.
        for key in score.keys():
            emit_prb[key] = math.exp(score[key] - max_score)
            expsum += emit_prb[key]
        for key in score.keys():
            emit_prb[key] /= expsum
            emit_prb[key] = math.log(emit_prb[key])
        return emit_prb

    def ViterbiDecode(self, sentence):
        N = len(sentence)  # length of the sentence
        prb = 0.
        prb_max = 0.
        toward = list()
        back = list()

        # get the feature Vector of every single character
        features = self.GetFeature(sentence)
        vec = self.Feature2Vec(features)

        for i in range(N):
            toward.append({})
            back.append({})
            for j in self.state:
                toward[i][j] = float('-inf')
                back[i][j] = ' '

        # run viterbi
        score = self.classifiy_score(vec[0])
        emit_prb = self.getEmitPrb(score)
        # print emit_prb
        for s in self.state:
            toward[0][s] = self.init_prb[s] + emit_prb[s]
            back[0][s] = 'end'
        # toward algorithm
        for t in range(1, N):
            score = self.classifiy_score(vec[t])
            # print score
            emit_prb = self.getEmitPrb(score)
            for s in self.state:
                prb = float('-inf')
                prb_max = float('-inf')
                state_max = 'S'
                for i in self.state:
                    prb = toward[t - 1][i] + self.trans_prb[i][s] + emit_prb[s]
                    if prb > prb_max:
                        prb_max = prb
                        state_max = i
                toward[t][s] = prb_max
                back[t][s] = state_max
        # backward algorithm to get the best tag sequence
        index = N - 1
        taglist = []
        prb_max = float('-inf')
        state_max = ''
        for s in self.state:
            prb = toward[N - 1][s]
            if prb > prb_max:
                prb_max = prb
                state_max = s
        taglist.append(state_max)
        while index >= 1:
            pre_state = back[index][taglist[0]]
            taglist.insert(0, pre_state)
            index -= 1
        if taglist[-1] == 'B':
            taglist[-1] = 'S'
        elif taglist[-1] == 'M':
            taglist[-1] == 'E'
        return taglist

    def GetFeature(self, sent):
        """
        get feature for every single character
        return a list of features
        """
        features = []
        for i in range(len(sent)):
            left2 = sent[i - 2] if i - 2 >= 0 else '#'
            left1 = sent[i - 1] if i - 1 >= 0 else '#'
            mid = sent[i]
            right1 = sent[i + 1] if i + 1 < len(sent) else '#'
            right2 = sent[i + 2] if i + 2 < len(sent) else '#'

            # get dictionary imformation
            if self.dict.dic.has_key(mid):
                MWL = str(self.dict.dic[mid][0])
                t0 = self.dict.dic[mid][1]
            else:
                MWL = '0'
                t0 = '#'

            feat = [left2, left1, mid, right1, right2, left2 + left1,
                    left1 + mid, mid + right1, right1 + right2, left1 + right1,
                    MWL + t0, left1 + t0, mid + t0, right1 + t0]
            features.append(feat)

        return features

    def Feature2Vec(self, feats):
        """
        get feature vector from feature
        the paramters feats mean is a list of features of every character
        """
        punctuation = [u'。', u'，', u'？', u'！', u'、', u'；', u'：', u'「', '」',
                       u'『', u'』', u'‘', u'’', u'“', u'”', u'（', u'）', u'〔',
                       u'〕', u'【', u'】', u'——', u'–', u'…', u'．', u'·', u'《',
                       u'》', u'〈', u'〉']
        featVecs = []
        for feat in feats:
            featVec = {}
            # if feat[2] in punctuation:
            #     featVec[0] = 1
            for it in range(len(feat)):
                if it < 5:
                    if self.unigram_feat_id.has_key(feat[it]):
                        key = self.unigram_feat_id[
                            feat[it]] + self.unigram_feat_num * it
                        featVec[key] = 1
                elif it < 10:
                    if self.bigram_feat_id.has_key(feat[it]):
                        key = self.bigram_feat_id[feat[it]]
                        key += self.unigram_feat_num * 5 + \
                            self.bigram_feat_num * (it - 5)
                        featVec[key] = 1
                else:
                    if self.dict_feat_id.has_key(feat[it]):
                        key = self.dict_feat_id[feat[it]]
                        key += self.unigram_feat_num * 5 + self.bigram_feat_num * \
                            5 + self.dict_feat_num * (it - 10)
                        featVec[key] = 1
#                        if key>self.dimension:
#                            self.dimension = key
            featVecs.append(featVec)

        return featVecs

    def getTag(self, wordlist):
        """get the tag for every char in the word"""
        taglist = []
        for word in wordlist:
            if len(word) == 1:
                taglist.append('S')
            else:
                taglist.append('B')
                for w in word[1:len(word) - 1]:
                    taglist.append('M')
                taglist.append('E')
        return taglist

    def tag2word(self, sentence, taglist):
        wordlist = []
        tmp = ''
        for i in range(len(taglist)):
            if taglist[i] == 'S':
                tmp = sentence[i]
                wordlist.append(tmp)
                tmp = ''
            elif taglist[i] == 'B':
                tmp += sentence[i]
            elif taglist[i] == 'M':
                tmp += sentence[i]
            else:
                tmp += sentence[i]
                wordlist.append(tmp)
                tmp = ''
        return wordlist

    def loadTestCorpus(self, corpus_file):
        print "Loading Test Corpus data",
        input_data = codecs.open(corpus_file, 'r', 'utf-8')
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                self.corpus_num += 1
            if self.corpus_num % 1000 == 0 and self.corpus_num != 0:
                print '.',
            wordlist = rawText.split()
            sentence = "".join(wordlist)
            self.corpus.append(sentence)  # add to x, i.d. the corpus
        print "\nLoading Test Corpus done."

    def loadCorpus(self, corpus_file):
        print "Loading Corpus data",
        input_data = codecs.open(corpus_file, 'r', 'utf-8')
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                self.corpus_num += 1
            if self.corpus_num % 1000 == 0 and self.corpus_num != 0:
                print '.',
            wordlist = rawText.split()
            taglist = self.getTag(wordlist)
            self.tag.append(taglist)  # add to y, i.d. the tags list
            sentence = "".join(wordlist)
            self.corpus.append(sentence)  # add to x, i.d. the corpus
        print "\nLoading Corpus done."

    def pretreatment(self, train_file):
        print "The process of corpus Pretreatment",
        input_data = codecs.open(train_file, 'r', 'utf-8')
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                self.corpus_num += 1
            if self.corpus_num % 1000 == 0 and self.corpus_num != 0:
                print '.',
            wordlist = rawText.split()
            taglist = self.getTag(wordlist)
            self.tag.append(taglist)  # add to y, i.d. the tags list
            sentence = "".join(wordlist)
            self.corpus.append(sentence)  # add to x, i.d. the corpus
            self.init_prb[taglist[0]] += 1
            for t in range(1, len(taglist)):
                self.trans_prb[taglist[t - 1]][taglist[t]] += 1

            feats = self.GetFeature(sentence)
            # record the feats, allocate the id of feature
            for feat in feats:
                for it in range(len(feat)):
                    if it < 5:  # unigram feature
                        if not self.unigram_feat_id.has_key(feat[it]):
                            self.unigram_feat_num += 1
                            self.unigram_feat_id[
                                feat[it]] = self.unigram_feat_num
                    elif it < 10:  # bigram feature
                        if not self.bigram_feat_id.has_key(feat[it]):
                            self.bigram_feat_num += 1
                            self.bigram_feat_id[
                                feat[it]] = self.bigram_feat_num
                    else:  # dictionary information feature
                        if not self.dict_feat_id.has_key(feat[it]):
                            self.dict_feat_num += 1
                            self.dict_feat_id[feat[it]] = self.dict_feat_num

        # calculate the probability of tag
        initsum = sum(self.init_prb.values())
        for key in self.init_prb.keys():
            self.init_prb[key] = float(self.init_prb[key]) / initsum
        for x in self.trans_prb.keys():
            tmpsum = sum(self.trans_prb[x].values())
            for y in self.trans_prb[x].keys():
                self.trans_prb[x][y] = float(self.trans_prb[x][y]) / tmpsum
        self.dimension = self.unigram_feat_num * 5 + \
            self.bigram_feat_num * 5 + self.dict_feat_num * 4
        # calc the log probability
        for s in self.state:
            if self.init_prb[s] != 0.:
                self.init_prb[s] = math.log(self.init_prb[s])
            else:
                self.init_prb[s] = float('-inf')
            for j in self.state:
                if self.trans_prb[s][j] != 0.:
                    self.trans_prb[s][j] = math.log(self.trans_prb[s][j])
                else:
                    self.trans_prb[s][j] = float('-inf')
        print "\nProcess of pretreatment finished."
