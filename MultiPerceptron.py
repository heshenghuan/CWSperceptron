# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:49:02 2015

@author: heshenghuan
"""

import math
import random
import copy
import codecs
import datetime
from cPickle import dump
from cPickle import load

def calc_acc(labellist1, labellist2):
    samelist =[int(x == y) for (x, y) in zip(labellist1, labellist2)]
    accuracy = float((samelist.count(1)))/len(samelist)
    return accuracy

class MultiPerceptron(object):
    def __init__(self):
        self.label_set_list = list()
        self.weights = list()
        #self.best_weights = list()
        self.feat_size = 0
        self.sample_list = list()
        self.label_list = list()
        self.path = r"./"

    def sigmoid(self,x):
        return 1/(1+math.exp(-x/5000))
    
    def printinfo(self):
        print "sample size:", len(self.sample_list)
        print "label size:", len(self.label_list)
        print "label set size:", len(self.label_set_list)
        print "feat size:", self.feat_size

    def setSavePath(self,path):
        self.path = path
        
    def saveModel(self):
        output1 = open(self.path+r"label_set_list.pkl",'wb')
        dump(self.label_set_list, output1, -1)
        output1.close()
        output2 = open(self.path+r"weights.pkl",'wb')
        dump(self.weights, output2, -1)
        output2.close()
        #release the memory
        self.label_set_list = list()
        self.weights = list()
        self.best_weights = list()
        self.sample_list = list()
        self.label_list = list()
        
    def loadLabelSet(self):
        inputs = open(self.path+r"label_set_list.pkl",'rb')
        self.label_set_list = load(inputs)
        inputs.close()
        
    def loadWeights(self):
        inputs = open(self.path+r"weights.pkl",'rb')
        self.weights = load(inputs)
        inputs.close()
        
    def loadFeatSize(self, size, classNum):
        self.feat_size = size
        omega = []
        for i in xrange(classNum):
            omega.append(copy.deepcopy([0.]*(size+1)))
        self.weights = omega
    
    def printweights(self):
        for item in self.weights:
            print item
    
    def train_mini_batch(self, batch_num = 10, max_iter = 100, learn_rate = 0.01, delta_thrd = 0.00001, is_average = True):
        print '-'*60
        print "START TRAIN MINI BATCH:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size+1 # number of feature
        C = len(self.label_set_list)
        
        if batch_num > N:
            batch_num = N
        sample_list = self.sample_list
        #weights of each class
        omega = self.weights  #just use the weights, the old weights is saved on disk
        omega_sum  = []
        if is_average:
            for i in xrange(C):
                omega_sum.append(copy.deepcopy([0.]*M))
        rd = 0
        loss = 0.
        start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print start_clock
        delta = []
        for i in xrange(C):
            delta.append(copy.deepcopy([0.]*M))
        while rd < max_iter:
            #update omega
            for i in xrange(C):
                for j in xrange(M):
                    delta[i][j] = 0.
            time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch_list = []
            while(len(batch_list)<batch_num):
                index = random.randint(0,N-1)
                if index not in batch_list:
                    batch_list.append(index)
            for i in batch_list:
                sample = sample_list[i]
                result = [0.]*C
                for index in range(C):
                    result[index] = sum(omega[index][j]*sample[j] for j in sample.keys())
                max_label_id = result.index(max(result))                  
                max_label = self.label_set_list[max_label_id]
                real_label = self.label_list[i]
                real_label_id = self.label_set_list.index(real_label)
                if real_label != max_label:
                    for index in sample.keys():
                        delta[real_label_id][index] +=  learn_rate * sample[index]
                        delta[max_label_id][index] -=   learn_rate * sample[index]
            # weight update
            for l in range(C):
                for j in range(M):
                    omega[l][j] += delta[l][j]
                    if is_average:
                        omega_sum[l][j] += omega[l][j]
            
            error_count = 0
            loss = 0.
            for j in range(N):
                sample1 = sample_list[j]
                result = [0.]*C
                for index in range(C):
                    result[index] = sum(omega[index][p]*sample1[p] for p in sample1.keys())
                max_label_id1 = result.index(max(result))                  
                max_label1 = self.label_set_list[max_label_id1]
                real_label1 = self.label_list[j]
                real_label_id1 = self.label_set_list.index(real_label1)
                if real_label1 != max_label1:
                    error_count += 1
                    loss += (result[max_label_id1] - result[real_label_id1])
            acc = 1-error_count/float(N)
            loss /= N
            time2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print '\nIter:', rd, '\tCost:', '%.8f' % loss, '\tAcc:', '%.4f' % acc,"\tError: %4d" %error_count
            print 'start:\t',time1
            print "stop:\t",time2
            
            if loss < delta_thrd or acc>0.99:
                print "Reach the minimal loss value threshold!"
                break
            rd+=1

        if is_average:
            self.weights = [[omega_sum[l][j]/(rd) for j in range(M)] for l in range(C)]
        else:
            self.weights = omega
        stop_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "\nTraining process finished."
        print "start time:\t",start_clock
        print "stop time:\t",stop_clock
    
    def train_sgd(self, max_iter = 100, learn_rate = 1.0, delta_thrd = 0.00001, is_average = True):
        print '-'*60
        print "START TRAIN SGD:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size+1 # number of feature
        C = len(self.label_set_list)
        
        # convert to extend sample set
        sample_list_ext = self.sample_list
        #weights of each class
        omega = []
        for i in xrange(C):
            omega.append(copy.deepcopy([float(1)/N]*M))

        omega_sum  = copy.deepcopy(omega)
        rd = 0
        loss = 0.
        start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print start_clock
        while rd < max_iter*N:
            #update omega
            delta = []
            for i in xrange(C):
                delta.append(copy.deepcopy([0.]*M))
            time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            i = random.randint(0,N-1)
            sample = sample_list_ext[i]
            result = [0.]*C
            for index in range(C):
                result[index] = sum(omega[index][j]*sample[j] for j in sample.keys())
            max_label_id = result.index(max(result))                  
            max_label = self.label_set_list[max_label_id]
            real_label = self.label_list[i]
            real_label_id = self.label_set_list.index(real_label)
            # weight update
            if real_label != max_label:
                for index in sample.keys():
                    omega[real_label_id][index] +=  learn_rate * sample[index]
                    omega[max_label_id][index] -=   learn_rate * sample[index]
            
            error_count = 0
            loss = 0.
            for j in range(N):
                sample1 = sample_list_ext[j]
                result = [0.]*C
                for index in range(C):
                    result[index] = sum(omega[index][p]*sample1[p] for p in sample1.keys())
                max_label_id1 = result.index(max(result))                  
                max_label1 = self.label_set_list[max_label_id1]
                real_label1 = self.label_list[j]
                real_label_id1 = self.label_set_list.index(real_label1)
                if real_label1 != max_label1:
                    error_count += 1
                    loss += (result[max_label_id1] - result[real_label_id1])
            acc = 1-error_count/float(N)
            loss /= N
            time2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print '\nIter:', rd, '\tCost:', '%.8f' % loss, '\tAcc:', '%.4f' % acc,"\tError: %4d" %error_count
            print 'start:\t',time1
            print "stop:\t",time2
            
            if loss < delta_thrd :
                print "Reach the minimal loss value threshold!"
                break
            rd+=1
            if is_average:
                for l in range(C):
                    for j in range(M):
                        omega_sum[l][j] += omega[l][j]

        if is_average:
            self.weights = [[omega_sum[l][j]/(rd) for j in range(M)] for l in range(C)]
        else:
            self.weights = omega
        stop_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "\nTraining process finished."
        print "start time:\t",start_clock
        print "stop time:\t",stop_clock    
    
    def train_batch(self, max_iter = 100, delta_thrd = 0.00001, is_average = True):
        print '-'*60
        print "START TRAIN BATCH:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size+1 # number of feature
        C = len(self.label_set_list)
        
        # convert to extend sample set
        sample_list_ext = self.sample_list
        #weights of each class
        omega = []
        for i in xrange(C):
            omega.append(copy.deepcopy([0.]*M))
#            for j in xrange(M):
#                self.weights[i].append(0.)
        omega_sum  = copy.deepcopy(omega)
        learn_rate = 1
        rd = 0
        best_acc = 0.
        best_omega = []
        while rd < max_iter:
            error_count = 0
            delta = []
            for i in xrange(C):
                delta.append(copy.deepcopy([0.]*M))
            for i in xrange(N):
                sample = sample_list_ext[i]
                #print "sample size:", max(sample.keys())
                result = [0.]*C
                for index in xrange(C):
                    result[index] = sum(omega[index][j]*sample[j] for j in sample.keys())
                max_label_id = result.index(max(result))                  
                max_label = self.label_set_list[max_label_id]
                real_label = self.label_list[i]
                real_label_id = self.label_set_list.index(real_label)
                if real_label != max_label:
                    error_count += 1
                    for index in sample.keys():
#                        delta[real_label_id][index] = omega[real_label_id][index] + learn_rate * value
#                        delta[max_label_id][index] = omega[max_label_id][index] - learn_rate * value
                        delta[real_label_id][index] +=  learn_rate * sample[index]
                        delta[max_label_id][index] -=   learn_rate * sample[index]
            acc = 1-float(error_count)/N
            print 'Iter:', rd+1, '\tAcc: %.4f' %acc,"\tError: %4d" %error_count
            if acc > best_acc:
                best_acc = acc
                best_omega = copy.deepcopy(omega)
            # weight update
            for l in range(C):
                for j in range(M):
                    omega[l][j] += delta[l][j]
                    omega_sum[l][j] += omega[l][j]
            rd+=1
#            if acc >= 0.9900:
#                break

        if is_average:
            self.weights = [[omega_sum[l][j]/(rd*N) for j in range(M)] for l in range(C)]
        else:
            #self.weights = omega
            self.weights = best_omega
        print "Best Accuract: %.4f" %best_acc
        #self.best_weights = best_omega

    def train_random(self, max_iter = 100, learn_rate = 1, delta_thrd = 0.00001, is_average = True):
        print '-'*60
        print "START TRAIN SGD:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size+1 # number of feature
        C = len(self.label_set_list)
        
        # convert to extend sample set
        sample_list_ext = copy.deepcopy(self.sample_list)
        for i in range(N):
            sample_list_ext[i][0]=1    #add bias
        #weights of each class
        omega = []
        for i in xrange(C):
            omega.append(copy.deepcopy([float(1)/N]*M))

        omega_sum  = copy.deepcopy(omega)
        rd = 0
        loss = 0.
        loss_pre = 0.
        #best_acc = 0.
        start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print start_clock
        while rd < max_iter*N:
            #if rd%1000 == 0:
                #print rd,
            if rd%N == 0:
                loop = rd/N
                error_count = 0
                loss = 0.
                for j in range(N):
                    sample = sample_list_ext[j]
                    result = [0.]*C
                    for index in range(C):
                        result[index] = sum(omega[index][p]*sample[p] for p in sample.keys())
                    max_label_id = result.index(max(result))                  
                    max_label = self.label_set_list[max_label_id]
                    real_label = self.label_list[j]
                    real_label_id = self.label_set_list.index(real_label)
                    if real_label != max_label:
                        error_count += 1
                        loss += (result[max_label_id] - result[real_label_id])
                acc = 1-error_count/float(N)
                loss /= N
                time2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print '\nIter:', loop, '\tCost:', '%.8f' % loss, '\tAcc:', '%.4f' % acc,"\tError: %4d" %error_count
                print 'start:\t',
                if rd!=0:
                    print time1
                else:
                    print start_clock
                print "stop:\t",time2
                if rd!=0 and loss_pre - loss < delta_thrd and loss_pre >= loss:
                    print "Reach the minimal loss value decrease!"
                    break
                loss_pre = loss
                time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                if acc > best_acc:
#                    best_acc = acc
#                    self.best_weights = copy.deepcopy(omega)
            
            #update omega
            i = random.randint(0,N-1)
            sample = sample_list_ext[i]
            result = [0.]*C
            for index in range(C):
                result[index] = sum(omega[index][j]*sample[j] for j in sample.keys())
            max_label_id = result.index(max(result))                  
            max_label = self.label_set_list[max_label_id]
            real_label = self.label_list[i]
            real_label_id = self.label_set_list.index(real_label)
            # weight update
            if real_label != max_label:
                for index in sample.keys():
                    omega[real_label_id][index] +=  learn_rate * sample[index]
                    omega[max_label_id][index] -=   learn_rate * sample[index]
            #sum for average
            for l in range(C):
                for j in range(M):
                    omega_sum[l][j] += omega[l][j]
            rd+=1

        if is_average:
            self.weights = [[omega_sum[l][j]/(rd) for j in range(M)] for l in range(C)]
        else:
            if rd >= max_iter*N:
                self.weights = self.best_weights
            else:
                self.weights = omega
        stop_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "start time:\t",start_clock
        print "stop time:\t",stop_clock

    def scoreout(self,sample_test):
        sample_test_ext = sample_test
        C = len(self.weights)
        score = {}
        for l in range(C):
            label = self.label_set_list[l]
            score[label] = sum(self.weights[l][j]*sample_test_ext[j] for j in sample_test_ext.keys())
        return score
    
    def probout(self, sample_test):
        sample_test_ext = sample_test
        C = len(self.weights)
        score = {}
        prb = {}
        for l in range(C):
            label = self.label_set_list[l]
            score[label] = sum(self.weights[l][j]*sample_test_ext[j] for j in sample_test_ext.keys())
            prb[label] = self.sigmoid(score[label])
        return prb

    def classify(self, sample_test):
        sample_test_ext = sample_test
        C = len(self.weights)
        weighted_sum = [0.]*C
        for l in range(C):
            weighted_sum[l] = sum(self.weights[l][j]*sample_test_ext[j] for j in sample_test_ext.keys())
        max_label_id = weighted_sum.index(max(weighted_sum))
        max_label = self.label_set_list[max_label_id]
        return max_label
    
    def batch_classify(self, sample_list_test):
        label_list_test = []
        for sample_test in sample_list_test:
            label_list_test.append(self.classify(sample_test))
        return label_list_test
    
    def read_train_file(self, filepath):
        """
        make traing set from file 
        return sample_set, label_set
        """
        data = codecs.open(filepath, 'r')
        for line in data.readlines():
            val = line.strip().split('\t')
            self.label_list.append(val[0])
#            max_index = 0
            sample_vec = {}
            val = val[-1].split(" ")
            for i in range(0, len(val)):
                [index, value] = val[i].split(':')
                sample_vec[int(index)] = float(value)
            self.sample_list.append(sample_vec)
        self.label_set_list = list(set(self.label_list))
