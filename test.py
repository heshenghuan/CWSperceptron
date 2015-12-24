#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12:51:52 2015-11-09

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

from CWSPv4 import CWSPerceptron as CWSP
import Dict

if __name__ == '__main__':
    # # create an instance of CWSPerceptron
    # cws = CWSP()
    # # pre deal the corpus file
    # cws.pretreatment(r'./icwb2-data/training/pku_training.utf8')
    # # set save path of models file, it is a directory
    # cws.setSavePath(r'./models/pku/')

    # # print information
    # print u"corpus number:        ", cws.corpus_num
    # print u"unigram number:       ", cws.unigram_feat_num
    # print u"bigram number:        ", cws.bigram_feat_num
    # print u"Features dimension:   ", cws.dimension
    # print u"Initial probability:"
    # for i in cws.init_prb:
    #     print i, cws.init_prb[i]
    # print u"trans probability:"
    # for i in cws.trans_prb:
    #     print i, cws.trans_prb[i]

    # # make training data
    # count = cws.makeLibSvmData(r'./training/pku/data',-1)
    # print 'generate',count,'training data file.'

    # # save the model to release memory
    # cws.saveModel()

    # # training process
    # cws.train(r'./training/pku/data',count,5000,500,1,0.1,False)

    # # training finished, load models(labelset, prbs)
    # cws.loadModel()
    # # load test file
    # cws.loadCorpus(r'./icwb2-data/testing/pku_test.utf8')
    # # multiperceptron load model
    # cws.perceptron.loadFeatSize(cws.dimension,4)
    # cws.perceptron.loadLabelSet()
    # cws.perceptron.loadWeights()
    # # do segmentation
    # cws.segmentation(r"pku_seg.utf8")
    # del cws
    d = Dict.Dict()
    d.loadDict(r'dict.utf8')
    d.appendDict(r'test_gold_dict.utf8')
    d.saveDict(r'dict_with_testinfo.utf8')
