# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:21:20 2015

@author: heshenghuan
"""

import codecs
import copy
import math
from cPickle import dump
from cPickle import load

class CWSPerceptron:
    def __init__(self):
        self.corpus = list()  #save the corpus for training
        self.tag = list()     #the tag of corpus
        self.corpus_featVec = list()
        self.corpus_num = 0
        self.state = ['B','M','E','S']
        self.alpha = {'B':dict(), 'M':dict(), 'E':dict(), 'S':dict()}
        self.alpha_norm = {'B':0., 'M':0., 'E':0., 'S':0.}
        self.init_prb = {'B':0, 'M':0, 'E':0, 'S':0}
        self.trans_prb = {
                          'B':{'B':0, 'M':0, 'E':0, 'S':0}, 
                          'M':{'B':0, 'M':0, 'E':0, 'S':0}, 
                          'E':{'B':0, 'M':0, 'E':0, 'S':0}, 
                          'S':{'B':0, 'M':0, 'E':0, 'S':0}
                          }
        self.feat_num = 0
        self.feat_id = dict()
    
    def evaluate(self, corpus = 200):
        error_count = 0
        tagnums = sum([len(item) for item in self.tag[0:corpus]])
        for i in range(corpus):
            tag = self.ViterbiDecode(self.corpus[i])
            print 'y:',self.tag[i]
            print 'p:',tag
            for index in range(len(tag)):
                pre = tag[index]
                #print self.tag[j]
                real = self.tag[i][index]
                #print pre, real
                if pre != real:
                    error_count += 1
        return 1-float(error_count)/tagnums
    
    def train(self, max_corpus = -1,max_iter=200, learn_rate = 1.0):
        # initial the averaged perceptron vector
        for i in self.alpha.keys():
            for j in range(1,self.feat_num+1):
                self.alpha[i][j] = 0.
        #print self.alpha
        alpha_sum = copy.deepcopy(self.alpha)
        
        if max_corpus == -1:
            max_corpus = self.corpus_num
        
        tagnums = sum([len(item) for item in self.tag[0:max_corpus]])
        #strat train
        print "\nSTART TRIAN THE PERCEPTRON:"
        for i in range(max_iter):
            # total max_iter times iteration
            print "\nIter :", i+1
            error_count = 0
            for j in range(max_corpus):
                if (j+1)%100 == 0:
                    print j+1,
                tagz = self.decode(j)
#                print 'Z:',tagz
#                print "Y:",self.tag[j]
                #raw_input("\ninput to continue")
                feat_vec_Y = self.corpus_featVec[j]
                for index in range(len(tagz)):
                    pre = tagz[index]
                    #print self.tag[j]
                    real = self.tag[j][index]
                    #print pre, real
                    if pre != real:
                        error_count += 1
                        for key in feat_vec_Y.keys():
                            #print 'key', key
                            self.alpha[real][key] += learn_rate * feat_vec_Y[key]
                            self.alpha[pre][key] -= learn_rate * feat_vec_Y[key]
                for key in self.alpha.keys():
                    for d in self.alpha[key].keys():
                        if alpha_sum[key].has_key(d):
                            alpha_sum[key][d] += self.alpha[key][d]
#                        else:
#                            alpha_sum[key][d] = 0.
#                            alpha_sum[key][d] += self.alpha[key][d]
                #raw_input("\ninput to continue")
            print '\nIter:', i+1, 'is done.'
            print 'Acc:',1-float(error_count)/tagnums
        for key in self.alpha.keys():
            for d in self.alpha[key].keys():
                self.alpha[key][d] = alpha_sum[key][d]/(max_iter*max_corpus)
                self.alpha_norm[key] += pow(self.alpha[key][d],2)
            self.alpha_norm[key] = math.sqrt(self.alpha_norm[key])
        return 1
    
    def printstr(self, wordlist):
        for item in wordlist:
            print item
        print " "
    
    def decode(self, index):
        """
        the parameter is the features set of the sentence
        return a taglist
        """
        score = {}
        taglist = []
        sentence = self.corpus[index]
        features = self.GetFeature(sentence)
        vec = []
        for feat in features:
            featVec = {}
            for it in feat:
                if self.feat_id.has_key(it):
                    key = self.feat_id[it]
                    if featVec.has_key(key):
                        featVec[key] += 1
                    else:
                        featVec[key] = 1
            vec.append(featVec)
        for i in range(len(vec)):
            score = self.classifiy_score(vec[i])
            max_sum = 0.
            max_tag = ''
            for key in score.keys():
                if score[key] >= max_sum:
                    max_sum = score[key]
                    max_tag = key
            taglist.append(max_tag)
        return taglist
        
    def makelibsvmdata(self, output_file):
        output_data = codecs.open(output_file, 'w')
        for i in range(self.corpus_num):
            sentence = self.corpus[i]
            taglist = self.tag[i]
            features = self.GetFeature(sentence)
            vec = []
            for feat in features:
                featVec = {}
                for it in feat:
                    if self.feat_id.has_key(it):
                        key = self.feat_id[it]
                        if featVec.has_key(key):
                            featVec[key] += 1
                        else:
                            featVec[key] = 1
                vec.append(featVec)
            for j in range(len(taglist)):
                output_data.write(taglist[j])
                output_data.write('\t')
                for key in vec[j].keys():
                    output_data.write(str(key))
                    output_data.write(':')
                    output_data.write(str(vec[j][key]))
                    output_data.write(' ')
                output_data.write("\n")
        
        output_data.close()
        
    def classifiy_score(self, featureVec):
        score = {'B':0. ,'M':0. ,'E':0. ,'S':0.}
        for key in score.keys():
            for d in featureVec.keys():
                #print key, d
                score[key] += featureVec[d] * self.alpha[key][d]
        return score
        
    def getEmitPrb(self, vec, score):
        norm = 0.
        for key in vec.keys():
            norm += pow(vec[key],2)
        norm = math.sqrt(norm)
        
        emit_prb = {}
        for key in score.keys():
            cos_similar = score[key]/(norm * self.alpha_norm[key])
            emit_prb[key] = 1 - (math.acos(cos_similar)/math.pi)
        return emit_prb
    
    def ViterbiDecode(self,sentence):
        N = len(sentence) #length of the sentence
        prb = 0.
        prb_max = 0.
        toward = list()
        back = list()
        
        #get the feature Vector of every single character
        features = self.GetFeature(sentence)
        vec = []
        for feat in features:
            featVec = {}
            for it in feat:
                if self.feat_id.has_key(it):
                    key = self.feat_id[it]
                    if featVec.has_key(key):
                        featVec[key] += 1
                    else:
                        featVec[key] = 1
            vec.append(featVec)

        for i in range(N):
            toward.append({})
            back.append({})
            for j in self.state:
                toward[i][j] = 0.
                back[i][j] = ' '
         
        #run viterbi         
        #print vec[0]
        score = self.classifiy_score(vec[0])
        emit_prb = self.getEmitPrb(vec[0],score)
        for s in self.state:
            toward[0][s] = self.init_prb[s] * emit_prb[s]
            back[0][s] = 'end'
        #toward algorithm
        for t in range(1, N):
            score = self.classifiy_score(vec[t])
            emit_prb = self.getEmitPrb(vec[t],score)
            for s in self.state:
                prb = 0.
                prb_max = 0.
                for i in self.state:
                    prb = toward[t-1][i] * self.trans_prb[i][s] * emit_prb[s]
                    if prb > prb_max:
                        prb_max = prb
                        state_max = i
                toward[t][s] = prb_max
                back[t][s] = state_max
        #backward algorithm to get the best tag sequence
        #print toward
        index = N-1
        taglist = []
        prb_max = 0.
        state_max = ''
        for s in self.state:
            prb = toward[N-1][s]
            #print s, prb
            if prb > prb_max:
                prb_max = prb
                state_max = s
        taglist.append(state_max)
        print 'max state:',state_max
        self.printstr(back)
        while index >= 1:
            pre_state = back[index][taglist[0]]
            taglist.insert(0, pre_state)
            index -= 1

        return taglist

    def GetFeature(self, sent):
        features = []
        head = sent[0]
        tail = sent[-1]
        for i in range(len(sent)):
            left2=sent[i-2] if i-2 >=0 else '#'
            left1=sent[i-1] if i-1 >=0 else '#'
            mid=sent[i]
            right1=sent[i+1] if i+1<len(sent) else '#'
            right2=sent[i+2] if i+2<len(sent) else '#'
            feat=[left2,left1,mid,right1,right2,
                    left2+left1,left1+mid,mid+right1,right1+right2,head+tail]
            features.append(feat)
        
        return features
    
    def Feature2Vec(self, feats):
        featVec = {}
        for feat in feats:
            for it in feat:
                if self.feat_id.has_key(it):
                    key = self.feat_id[it]
                    if featVec.has_key(key):
                        featVec[key] += 1
                    else:
                        featVec[key] = 1
                else:
                    self.feat_num += 1
                    self.feat_id[it] = self.feat_num
                    self.alpha[self.feat_num] = 0.
                    featVec[self.feat_num] = 1
        return featVec
    
    def getTag(self, wordlist):
        taglist = []
        for word in wordlist:
            if len(word)==1:
                taglist.append('S')
            else:
                taglist.append('B')
                for w in word[1:len(word)-1]:
                    taglist.append('M')
                taglist.append('E')
        return taglist
    
    def tag2word(self, sentence,taglist):
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
    
    def preDeal(self, train_file):
        input_data = codecs.open(train_file, 'r', 'utf-8')
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                self.corpus_num += 1
            wordlist = rawText.split()
            taglist = []
            for word in wordlist:
                if len(word)==1:
                    taglist.append('S')
                else:
                    taglist.append('B')
                    for w in word[1:len(word)-1]:
                        taglist.append('M')
                    taglist.append('E')
            self.init_prb[taglist[0]] += 1
            
            for t in range(1, len(taglist)):
                self.trans_prb[taglist[t-1]][taglist[t]] += 1
            self.tag.append(taglist)        #add to y, i.d. the tags list
            sentence = "".join(wordlist)
            self.corpus.append(sentence)    #add to x, i.d. the corpus
            #print sentence
            feats = self.GetFeature(sentence)
            #record the feats, allocate the id of feature
            for feat in feats:
                for it in feat:
                    if not self.feat_id.has_key(it):
                        self.feat_num += 1
                        self.feat_id[it] = self.feat_num
                    else:
                        continue
            
        for i in range(len(self.corpus)):
            feat = self.GetFeature(self.corpus[i])
            vec = self.Feature2Vec(feat)
            self.corpus_featVec.append(vec)
        initsum = sum(self.init_prb.values())
        for key in self.init_prb.keys():
            self.init_prb[key] = float(self.init_prb[key])/initsum
        for x in self.trans_prb.keys():
            tmpsum = sum(self.trans_prb[x].values())
            for y in self.trans_prb[x].keys():
                self.trans_prb[x][y] = float(self.trans_prb[x][y])/tmpsum

def Gen(sent):
    features = []
    head = sent[0]
    tail = sent[-1]
    for i in range(len(sent)):
        left2=sent[i-2] if i-2 >=0 else '#'
        left1=sent[i-1] if i-1 >=0 else '#'
        mid=sent[i]
        right1=sent[i+1] if i+1<len(sent) else '#'
        right2=sent[i+2] if i+2<len(sent) else '#'
        feat=[left2,left1,mid,right1,right2,
              left2+left1,left1+mid,mid+right1,right1+right2,head+tail]
        features.append(feat)
    return features

if __name__ == '__main__':
    cws = CWSPerceptron()
    cws.preDeal(r'.\\training\\pku_training.utf8')
    #inputs = open(r"cwsp.pkl",'rb')
    #cws = load(inputs)
    print u"语料数量：", cws.corpus_num
    print u"特征数量：",cws.feat_num
    print u"初始概率："
    for i in cws.init_prb:
        print i, cws.init_prb[i]
    print u"转移概率："
    for i in cws.trans_prb: 
        print i, cws.trans_prb[i]
    cws.makelibsvmdata(r'pku_training.data')
    #cws.train(2000,10,1.0)
    #taglist = cws.ViterbiDecode(cws.corpus[3])
    output = open(r"cwsp.pkl",'wb')
    dump(cws, output, -1)
    output.close()
    
#    error_count = 0
#    tagnums = sum(len(s) for s in cws.corpus[0:100])
#    for i in range(100):
#        tag = cws.decode(i)
#        for index in range(len(tag)):
#            pre = tag[index]
#            #print self.tag[j]
#            real = cws.tag[i][index]
#            #print pre, real
#            if pre != real:
#                error_count += 1
#    print 1-float(error_count)/tagnums
    #print 'alpha:',cws.alpha
#    text = [u"现在",u"的",u"网络",u"用语",u"变化",u"太",u"快",u"。"]
#    taglist = []
#    for word in text:
#        if len(word)==1:
#            taglist.append('S')
#        else:
#            taglist.append('B')
#            for w in word[1:len(word)-1]:
#                taglist.append('M')
#            taglist.append('E')
#
#    print "".join(taglist)
#    out = ''.join([i for i in text])
    #wordlist = cws.decode(out,16)
#    print out
#    wordlist = cws.tag2word(out, taglist)
#    
#    for word in text:
#        print word,
#    print
#    
#    for word in wordlist:
#        print word,
    
    #feats = Gen(out)
    #vec = cws.Feature2Vec(feats)
    #print len(feats)
    #print vec
#    text1 = [u"现在",u"的",u"网络",u"用语",u"变化",u"太快",u"。"]
#    taglist1 = []
#    for word in text1:
#        if len(word)==1:
#            taglist1.append('S')
#        else:
#            taglist1.append('B')
#            for w in word[1:len(word)-1]:
#                taglist1.append('M')
#            taglist1.append('E')
#
#    print "".join(taglist1)
#    out1 = ''.join([i for i in text1])
#
#    print out1
#    feats1 = Gen(out1)
#    vec1 = cws.Feature2Vec(feats1, taglist1)
#    print len(feats1)
#    print vec1
#    print cws.IsTagSame(taglist, taglist1)
#    taglist1 = copy.deepcopy(taglist)
#    print cws.IsTagSame(taglist,taglist1)