# -*- coding: utf-8 -*-
"""
Created on Thu Aug 06 15:31:27 2015

@author: heshenghuan
"""

"""
CWS perceptron version 0.5
using featrue template 10+punc+dict
"""

import codecs
import math
from MultiPerceptron import MultiPerceptron as MP
from Dict import Dict
from cPickle import dump
from cPickle import load

class CWSPerceptron:
    def __init__(self):
        self.corpus = list()  #save the corpus for training
        self.tag = list()     #the tag of corpus
        self.corpus_num = 0
        self.state = ['B','M','E','S']
        self.perceptron = MP()
        self.init_prb = {'B':0, 'M':0, 'E':0, 'S':0}
        self.trans_prb = {
                          'B':{'B':0, 'M':0, 'E':0, 'S':0}, 
                          'M':{'B':0, 'M':0, 'E':0, 'S':0}, 
                          'E':{'B':0, 'M':0, 'E':0, 'S':0}, 
                          'S':{'B':0, 'M':0, 'E':0, 'S':0}
                          }
        self.dimension = 0
        self.unigram_feat_num = 0
        self.unigram_feat_id = {}
        self.bigram_feat_num = 0
        self.bigram_feat_id = {}
        self.dict = Dict()
        #self.dict_feat_num = 0
        #self.dict_feat_id = {}
    
    def saveModel(self):
        print "Saving the unigram&bigram infomation......"
        output1 = open(r"bigram_feat_id.pkl",'wb')
        dump(self.bigram_feat_id, output1, -1)
        output1.close()
        output2 = open(r"unigram_feat_id.pkl",'wb')
        dump(self.unigram_feat_id, output2, -1)
        output2.close()
        
        #release the memory
        self.unigram_feat_id = []
        self.bigram_feat_id = []
        self.corpus = []
        self.tag = []
        print "Saving the inital prb & trans prb infomation....."
        output1 = open(r"init_prb.pkl",'wb')
        dump(self.init_prb, output1, -1)
        output1.close()
        output2 = open(r"trans_prb.pkl",'wb')
        dump(self.trans_prb, output2, -1)
        output2.close()
        print "Saving process done."
        
    def loadModel(self):
        print "Loading the unigram&bigram infomation......"
        inputs = open(r"bigram_feat_id.pkl",'rb')
        self.bigram_feat_id = load(inputs)
        self.bigram_feat_num = len(self.bigram_feat_id)
        inputs.close()
        inputs1 = open(r"unigram_feat_id.pkl",'rb')
        self.unigram_feat_id = load(inputs1)
        self.unigram_feat_num = len(self.unigram_feat_id)
        inputs1.close()
        #print "Loading process done."
        print "Loading the prb infomation......"
        inputs = open(r"init_prb.pkl",'rb')
        self.init_prb = load(inputs)
        inputs.close()
        inputs1 = open(r"trans_prb.pkl",'rb')
        self.trans_prb = load(inputs1)
        inputs1.close()
        print "Loading process done."
        self.dimension = self.unigram_feat_num*5 + self.bigram_feat_num*9
        
    def loadDict(self,dictfile):
        self.dict.loadDict(dictfile)
        
    def saveDict(self,outfile):
        self.dict.saveDict(outfile)
        
    def readDict(self,dictfile):
        self.dict.readDict(dictfile)
    
    def evaluate(self, corpus = 200):
        error_count = 0
        tagnums = sum([len(item) for item in self.tag[0:corpus]])
        for i in range(corpus):
            tag = self.ViterbiDecode(self.corpus[i])
            #print 'y:',self.tag[i]
            #print 'p:',tag
            for index in range(len(tag)):
                pre = tag[index]
                #print self.tag[j]
                real = self.tag[i][index]
                #print pre, real
                if pre != real:
                    error_count += 1
        return 1-float(error_count)/tagnums
    
    def segmentation(self, outfile):
        """Doing segmentation work"""
        output = codecs.open(outfile,'w','utf-8')
        for i in range(self.corpus_num):
            taglist = self.ViterbiDecode(self.corpus[i])
            wordlist = self.tag2word(self.corpus[i],taglist)
            for j in range(len(wordlist)):
                output.write(wordlist[j])
                output.write(' ')
            output.write("\n")
        output.close()
    
    def train(self,trainfile,filenum,batch_num=100, max_iter=200, learn_rate = 0.01, delta_thrd = 0.00001, is_average = True):
        """Training the perceptron model"""
        #self.makelibsvmdata(r'train.data',max_corpus)
        print "Start training process."
        self.perceptron.loadFeatSize(self.dimension,len(self.state))
        for i in range(1, filenum):
            if i>1:
                self.perceptron.loadWeights()
                self.perceptron.loadLabelSet()
            print "reading training file",i
            self.perceptron.read_train_file(trainfile+str(i))
            self.perceptron.printinfo()
            #self.perceptron.train_mini_batch(1000,500,0.01,10,False)
            self.perceptron.train_mini_batch(batch_num, max_iter, learn_rate, delta_thrd, is_average)
            self.perceptron.saveModel()
        print "Training process done."
        print "Multi-class Perceptron Model had been saved."
    
    def printstr(self, wordlist):
        for item in wordlist:
            print item
        print " "
        
    def makeLibSvmData(self, output_file,corpus_num = -1):
        """From original corpus make usable data for perceptron"""
        print "Making training data.",
        filecount = 1
        output_data = codecs.open(output_file+str(filecount), 'w')
        if corpus_num == -1:
            corpus_num = self.corpus_num
        s_count = 0
        for i in range(corpus_num):
            taglist = self.tag[i]
            features = self.GetFeature(self.corpus[i])
            vec = self.Feature2Vec(features)
            for j in range(len(taglist)):
                output_data.write(taglist[j])
                #output_data.write(str(self.state.index(taglist[j])))
                output_data.write('\t')
                keyset = list(vec[j].keys())
                keyset = sorted(keyset)
                for key in keyset:
                    output_data.write(str(key))
                    output_data.write(':')
                    output_data.write(str(vec[j][key]))
                    output_data.write(' ')
                output_data.write("\n")
                s_count += 1
                if s_count%100000==0:
                    filecount += 1
                    output_data.close()
                    output_data = codecs.open(output_file+str(filecount), 'w')
                    print '.',
        output_data.close()
        print "\nMaking training data finished.Totally",s_count,"samples."
        return filecount
        
    def classifiy_score(self, featureVec):
        return self.perceptron.scoreout(featureVec)
        #return self.perceptron.probout(featureVec)
        
    def getEmitPrb(self, score):
        """get emits_prb use softmax function"""
        max_score = max(score.values())
        emit_prb = {}
        expsum = 0.
        for key in score.keys():
            emit_prb[key] = math.exp(score[key]-max_score)
            expsum += emit_prb[key]
        for key in score.keys():
            emit_prb[key] /= expsum
        return emit_prb
    
    def ViterbiDecode(self,sentence):
        """Viterbi decode algorithm"""
        N = len(sentence) #length of the sentence
        prb = 0.
        prb_max = 0.
        toward = list()
        back = list()
        
        #get the feature Vector of every single character
        features = self.GetFeature(sentence)
        vec = self.Feature2Vec(features)

        for i in range(N):
            toward.append({})
            back.append({})
            for j in self.state:
                toward[i][j] = float('-inf')
                back[i][j] = ' '
         
        #run viterbi         
        score = self.classifiy_score(vec[0])
        emit_prb = self.getEmitPrb(score)
        #print emit_prb
        for s in self.state:
            #toward[0][s] = self.init_prb[s] * emit_prb[s]
            if self.init_prb[s] != 0. and emit_prb[s] != 0.:
                toward[0][s] = math.log(self.init_prb[s]) + math.log(emit_prb[s])
            back[0][s] = 'end'
        #toward algorithm
        for t in range(1, N):
            score = self.classifiy_score(vec[t])
            #print score
            emit_prb = self.getEmitPrb(score)
            for s in self.state:
                prb = float('-inf')
                prb_max = float('-inf')
                for i in self.state:
                    #prb = toward[t-1][i] * self.trans_prb[i][s] * emit_prb[s]
                    if self.trans_prb[i][s] != 0. and emit_prb[s]!=0.:
                        prb = toward[t-1][i] + math.log(self.trans_prb[i][s]) + math.log(emit_prb[s])
                    if prb > prb_max:
                        prb_max = prb
                        state_max = i
                toward[t][s] = prb_max
                back[t][s] = state_max
        #backward algorithm to get the best tag sequence
        index = N-1
        taglist = []
        prb_max = float('-inf')
        state_max = ''
        for s in self.state:
            prb = toward[N-1][s]
            #print s, prb
            if prb > prb_max:
                prb_max = prb
                state_max = s
        taglist.append(state_max)
        while index >= 1:
            pre_state = back[index][taglist[0]]
            taglist.insert(0, pre_state)
            index -= 1
        if taglist[-1]=='B':
            taglist[-1]='S'
        elif taglist[-1]=='M':
            taglist[-1]=='E'
        return taglist

    def GetFeature(self, sent):
        """
        get feature for every single character
        return a list of features
        """
        features = []
        for i in range(len(sent)):
            left2=sent[i-2] if i-2 >=0 else '#'
            left1=sent[i-1] if i-1 >=0 else '#'
            mid=sent[i]
            right1=sent[i+1] if i+1<len(sent) else '#'
            right2=sent[i+2] if i+2<len(sent) else '#'
            #print self.dict.dic.has_key(mid),
            if self.dict.dic.has_key(mid):
                MWL = str(self.dict.dic[mid][0])
                t0 = self.dict.dic[mid][1]
                #print MWL,t0
            else:
                MWL = '0'
                t0 = '#'
            #print MWL,t0
            feat=[left2,left1,mid,right1,right2,
                  left2+left1,left1+mid,mid+right1,
                  right1+right2,left1+right1,MWL+t0,
                  left1+t0,mid+t0,right1+t0]
            #feat=[left1,mid,right1,left1+mid,mid+right1]
            features.append(feat)
        
        return features
    
    def Feature2Vec(self, feats):
        """
        get feature vector from feature
        the paramters feats mean is a list of features of every character
        """
        punctuation = [u'。',u'，',u'？',u'！',u'、',u'；',u'：',u'「','」',
                       u'『',u'』',u'‘',u'’',u'“',u'”',u'（',u'）',u'〔',
                       u'〕',u'【',u'】',u'——',u'–',u'…',u'．',u'·',u'《',
                       u'》',u'〈',u'〉']
        featVecs = []
        for feat in feats:
            featVec = {}
            if feat[2] in punctuation:
                featVec[0] = 1
            for it in range(len(feat)):
                if it < 5:
                    if self.unigram_feat_id.has_key(feat[it]):
                        key = self.unigram_feat_id[feat[it]]+self.unigram_feat_num*it
                        featVec[key] = 1
                else:
                    if self.bigram_feat_id.has_key(feat[it]):
                        key = self.bigram_feat_id[feat[it]]
                        key += self.unigram_feat_num*5 + self.bigram_feat_num*(it-5)
                        featVec[key] = 1
#                        if key>self.dimension:
#                            self.dimension = key
            featVecs.append(featVec)                    
                
        return featVecs
    
    def getTag(self, wordlist):
        """get the tag for every char in the word"""
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
        """According the tag to rebuild word"""
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
    
    def loadTestCorpus(self,corpus_file):    
        print "Loading Test Corpus data",
        input_data = codecs.open(corpus_file, 'r', 'utf-8')
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                self.corpus_num += 1
            if self.corpus_num%1000 == 0 and self.corpus_num !=0:
                print '.',
            wordlist = rawText.split()
            sentence = "".join(wordlist)
            self.corpus.append(sentence)    #add to x, i.d. the corpus
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
            if self.corpus_num%1000 == 0 and self.corpus_num !=0:
                print '.',
            wordlist = rawText.split()
            taglist = self.getTag(wordlist)
            self.tag.append(taglist)        #add to y, i.d. the tags list
            sentence = "".join(wordlist)
            self.corpus.append(sentence)    #add to x, i.d. the corpus
        print "\nLoading Corpus done."
    
    def pretreatment(self, train_file):
        """Pretreatment of corpus"""
        print "The process of corpus Pretreatment",
        input_data = codecs.open(train_file, 'r', 'utf-8')
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                self.corpus_num += 1
            if self.corpus_num%1000 == 0 and self.corpus_num !=0:
                print '.',
            wordlist = rawText.split()
            taglist = self.getTag(wordlist)
            self.tag.append(taglist)        #add to y, i.d. the tags list
            sentence = "".join(wordlist)
            self.corpus.append(sentence)    #add to x, i.d. the corpus
            self.init_prb[taglist[0]] += 1
            for t in range(1, len(taglist)):
                self.trans_prb[taglist[t-1]][taglist[t]] += 1
            
            feats = self.GetFeature(sentence)
            #record the feats, allocate the id of feature
            for feat in feats:
                for it in range(len(feat)):
                    if it < 5:
                        if not self.unigram_feat_id.has_key(feat[it]):
                            self.unigram_feat_num += 1
                            self.unigram_feat_id[feat[it]] = self.unigram_feat_num
                    else:
                        if not self.bigram_feat_id.has_key(feat[it]):
                            self.bigram_feat_num += 1
                            self.bigram_feat_id[feat[it]] = self.bigram_feat_num
            
        #calculate the probability of tag
        initsum = sum(self.init_prb.values())
        for key in self.init_prb.keys():
            self.init_prb[key] = float(self.init_prb[key])/initsum
        for x in self.trans_prb.keys():
            tmpsum = sum(self.trans_prb[x].values())
            for y in self.trans_prb[x].keys():
                self.trans_prb[x][y] = float(self.trans_prb[x][y])/tmpsum
        self.dimension = self.unigram_feat_num*5 + self.bigram_feat_num*9
        print "\nProcess of pretreatment finished."
                
if __name__ == '__main__':
    cws = CWSPerceptron()
    cws.loadDict(r"dic.utf8")   #generate dict step must be the first step
    #cws.pretreatment(r'.\\FDU_NLPCC2015_Final\\train\\train-SEG.utf8')    
    cws.loadModel()
    print u"语料数量：\t", cws.corpus_num
    print u"unigram特征数量：\t",cws.unigram_feat_num
    print u"bigram特征数量：\t",cws.bigram_feat_num
    print u"特征空间维度： \t", cws.dimension
    print u"初始概率："
    for i in cws.init_prb:
        print i, cws.init_prb[i]
    print u"转移概率："
    for i in cws.trans_prb: 
        print i, cws.trans_prb[i]
    
    #count = cws.makeLibSvmData(r'.\\FDU_NLPCC2015_Final\\training\\dictdata',-1)
    #print 'generate',count,'training data file.'
    cws.saveModel()
    cws.train(r'.\\FDU_NLPCC2015_Final\\training\\dictdata',4,5000,500,1,0.1,False)        
    cws.loadModel()
    cws.loadCorpus(r'.\\FDU_NLPCC2015_Final\\test\\test-Gold-SEG.utf8')
    cws.perceptron.loadFeatSize(cws.dimension,4)
    cws.perceptron.loadLabelSet()
    cws.perceptron.loadWeights()
    cws.segmentation(r"FDU_seg.utf8")
    del cws
