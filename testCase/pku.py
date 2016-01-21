#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12:51:52 2015-11-09

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

from CWSPv6 import CWSPerceptron as CWSP

if __name__ == '__main__':
    cws = CWSP()
    cws.loadDict(r'./resources/dict.utf8')
    cws.setSavePath(r'./models/pku_tri/')
    # cws.pretreatment(r'./icwb2-data/training/pku_training.utf8')

    cws.loadModel()
    print u"corpus number:        ", cws.corpus_num
    print u"unigram number:       ", cws.unigram_feat_num
    print u"bigram number:        ", cws.bigram_feat_num
    print u"trigram number:       ", cws.trigram_feat_num
    print u"dict_feat number:     ", cws.dict_feat_num
    print u"type feature number:  ", cws.type_feat_num
    print u"Features dimension:   ", cws.dimension
    print u"Initial probability:"
    for i in cws.init_prb:
        print i, cws.init_prb[i]
    print u"trans probability:"
    for i in cws.trans_prb:
        print i, cws.trans_prb[i]

    # count = cws.makeLibSvmData(r'./training/pku/pku.dat', -1)
    # print 'generate', count, 'training data file.'
    # cws.saveModel()

    cws.loadCorpus(r'./icwb2-data/testing/pku_test.utf8')
    if cws.perceptron.loadModel(r'./models/pku_tri/pku.mdl'):
        print "Segmentation..."
        cws.segmentation(r"pku_seg.utf8")
        print "Done!"
    del cws
