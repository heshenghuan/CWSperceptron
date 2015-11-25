#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13:47:27 2015-3-15

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import math
import random
import codecs
import datetime
import time
import numpy as np
from cPickle import dump
from cPickle import load


CHANGED = "Updated on 13:18 2015-11-25"


def calc_acc(labellist1, labellist2):
    samelist = [int(x == y) for (x, y) in zip(labellist1, labellist2)]
    accuracy = float((samelist.count(1))) / len(samelist)
    return accuracy


class MultiPerceptron(object):
    """
    MultiPerceptron, a python implementation of multiclass perceptron
    algorithm.

    Using numpy.array to store matrixes.
    """

    def __init__(self):
        """
        Initialization function, returns an instance of MultiPerceptron with
        all the field empty.
        """
        self.label_set = []
        self.sample_list = []
        self.label_list = []
        self.Theta = None
        self.feat_size = 0
        self.class_num = 0
        self.path = r"./"  # the path used to store Theta matrix

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x / 5000))

    def printinfo(self):
        print "sample size:      ", len(self.sample_list)
        print "label size:       ", len(self.label_list)
        print "label set size:   ", len(self.label_set)
        print "feature dimension:", self.feat_size

    def setSavePath(self, path):
        """
        Set the model save path.
        """
        self.path = path

    def saveModel(self, path=r'./'):
        """
        Stores the model under given folder path.
        """
        if path == r"./":
            print "Using current directory(./) to save model."
        else:
            print "Storing model file under folder:", self.path, '.'

        output1 = open(self.path + r"label_set.pkl", 'wb')
        dump(self.label_set, output1, -1)
        output1.close()
        output2 = open(self.path + r"Theta.pkl", 'wb')
        dump(self.Theta, output2, -1)
        output2.close()
        # release the memory
        self.label_set = list()
        self.Theta = None
        self.sample_list = list()
        self.label_list = list()

    def loadLabelSet(self, label_set=None):
        """
        Loads label_set from file under given file path.

        If file does not exist, reports an IOError then returns False.

        If loads successed, returns True
        """
        if not label_set:
            print "Not given any file path, load label_set from default path."
            print "Please make sure corresponding file exist!"
            label_set = self.path + r"./label_set.pkl"

        try:
            inputs = open(label_set, 'rb')
            self.label_set = load(inputs)
            self.class_num = len(self.label_set)
            return True
        except IOError:
            print "Corresponding file \"label_set.pkl\" doesn\'t exist!"
            return False

    def loadTheta(self, Theta=None):
        """
        Loads Theta from file under given file path.

        If file does not exist, reports an IOError then returns False.

        If loads successed, returns True
        """
        if not Theta:
            print "Not given any file path, load Theta from default path."
            print "Please make sure corresponding file exist!"
            theta = self.path + r"./Theta.pkl"

        try:
            inputs = open(theta, 'rb')
            self.Theta = load(inputs)
            return True
        except IOError:
            print "Error:File does"
            print "Corresponding file \"Theta.pkl\" doesn\'t exist!"
            return False

    def loadFeatSize(self, size, classNum):
        """
        A combination of function setFeatSize, setClassNum and initTheta.
        In order to be compatible with old API of MultiPerceptron of mine.

        It has same result with the following sentences:
            >>> x.setFeatSize(feat_size)
            >>> x.setClassNum(classNum)
            >>> x.initTheta()
        """
        self.setFeatSize(size)
        self.setClassNum(classNum)
        flag = self.initTheta()
        return flag

    def setFeatSize(self, size=0):
        """
        Sets feature dimensions by the given size.
        """
        if size == 0:
            print "Warning: ZERO dimensions of feature will be set!"
            print "         This would causes some trouble unpredictable!"
            print "         Please make sure the dimension of feature is 0!"
        self.feat_size = size

    def setClassNum(self, classNum=0):
        """
        Sets number of label classies by given classNum.
        """
        if classNum == 0:
            print "Warning: ZERO class of samples will be set!"
            print "         This would causes some trouble unpredictable!"
            print "         Please make sure the number of classies is 0!"
        self.class_num = classNum

    def initTheta(self):
        """
        Initializes the Theta matrix.

        If the dimension of feature and number of classies are not be set, or
        are 0, this function will not Initializes Theta.

        Initialization successed, returns True. If not, returns False.
        """
        if self.feat_size != 0 and self.class_num != 0:
            self.Theta = np.zeros((self.class_num, self.feat_size + 1))
            return True
        else:
            print "Error: The dimension of feature and number of classies can"
            print "       not be ZERO!"
            return False

    def __getSampleVec(self, sample):
        """
        Returns a row vector by 1*(n+1).
        """
        sample_vec = np.zeros((1, self.feat_size + 1))
        sample_vec[0, 0] = 1.0
        for i in sample.keys():
            sample_vec[0][i] = sample[i]

        return sample_vec

    def predict(self, sample):
        """
        Returns the predict vector.
        """
        X = sample.T
        pred = []
        for j in range(self.class_num):
            pred.append(np.dot(self.Theta[j, :], X)[0])
        # return normalize(pred)
        return pred

    def predict2(self, sample):
        """
        Returns the predict vector, this function uses the raw sample to
        calculate while the predict<function> uses vector to represent the
        sample.
        """
        pred = []
        C = self.class_num
        for i in range(C):
            score = sum(self.Theta[i, key]*val for key, val in sample.items())
            pred.append(score)

        return pred

    def __getCost(self):
        """
        Returns the cost function value by using current Theta.
        """
        N = len(self.sample_list)    # number of sample

        error_count = 0
        loss = 0.
        for j in range(N):
            sample = self.sample_list[j]
            # result = self.predict(self.__getSampleVec(sample))
            result = self.predict2(sample)
            pred_id = result.index(max(result))
            pred = self.label_set[pred_id]
            label = self.label_list[j]
            label_id = self.label_set.index(label)
            if label != pred:
                error_count += 1
                loss += (result[pred_id] - result[label_id])

        loss /= N
        return (error_count, loss)

    def train_mini_batch(self, batch_num=100, max_iter=100, learn_rate=0.01,
                         delta_thrd=0.01, is_average=True):
        print '-' * 60
        print "START TRAIN MINI BATCH:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size + 1  # number of feature

        if batch_num > N:
            batch_num = N
        sample_list = self.sample_list
        # Theta of each class
        # just use the Theta, the old Theta is saved on disk
        omega = self.Theta
        omega_sum = np.zeros(omega.shape)
        rd = 1
        last = 0
        flag = 0
        start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print start_clock
        while rd <= max_iter:
            delta = np.zeros(omega.shape)
            # time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch_list = []
            while(len(batch_list) < batch_num):
                index = random.randint(0, N - 1)
                if index not in batch_list:
                    batch_list.append(index)

            for i in batch_list:
                result = self.predict2(sample_list[i])
                pred_id = result.index(max(result))
                pred = self.label_set[pred_id]
                label = self.label_list[i]
                label_id = self.label_set.index(label)
                if label != pred:
                    for key, val in sample_list[i].items():
                        delta[label_id, key] += learn_rate * val
                        delta[pred_id, key] -= learn_rate * val

            # weight update
            omega += delta
            omega_sum += omega

            # Get error & cost function value
            error_count, loss = self.__getCost()
            acc = 1 - error_count / float(N)
            # loss /= N
            # time2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print 'Iter:', rd, '\tCost:', '%.8f' % loss,
            print '\tAcc:', '%.4f' % acc, "\tError: %4d" % error_count
            # print 'start:\t', time1
            # print "stop:\t", time2

            if rd > 1:
                if last - loss < delta_thrd and last >= loss:
                    print "Reach the minimal loss value threshold!"
                    break
            rd += 1
            last = loss

        if is_average:
            self.Theta = omega_sum/(rd*batch_num)
        else:
            self.Theta = omega
        stop_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "\nTraining process finished."
        print "start time:\t", start_clock
        print "stop time:\t", stop_clock

    def train_batch(self, max_iter=100, learn_rate=0.01, delta_thrd=0.01,
                    is_average=True):
        print '-' * 60
        print "START TRAIN BATCH:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size + 1  # number of feature

        sample_list = self.sample_list
        omega = self.Theta
        omega_sum = np.zeros(omega.shape)
        rd = 1
        last = 0
        flag = 0
        start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print start_clock
        while rd <= max_iter:
            error_count = 0
            loss = 0.
            delta = np.zeros(omega.shape)
            # time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i in xrange(N):
                result = self.predict2(sample_list[i])
                pred_id = result.index(max(result))
                pred = self.label_set[pred_id]
                label = self.label_list[i]
                label_id = self.label_set.index(label)
                if label != pred:
                    error_count += 1
                    loss += (result[pred_id] - result[label_id])
                    for key, val in sample_list[i].items():
                        delta[label_id, key] += learn_rate * val
                        delta[pred_id, key] -= learn_rate * val

            acc = 1 - float(error_count) / N
            loss /= error_count if error_count != 0 else 1
            # time2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print 'Iter:', rd, '\tCost:', '%.8f' % loss,
            print '\tAcc:', '%.4f' % acc, "\tError: %4d" % error_count
            # print 'start:\t', time1
            # print "stop:\t", time2

            # weight update
            omega += delta
            omega_sum += omega
            if rd > 1:
                if last - loss < delta_thrd and last >= loss:
                    print "Reach the minimal loss value threshold!"
                    break
            rd += 1
            last = loss

        if is_average:
            self.Theta = omega_sum/(rd*N)
        else:
            # self.Theta = omega
            self.Theta = omega
        stop_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "\nTraining process finished."
        print "start time:\t", start_clock
        print "stop time:\t", stop_clock

    def train_sgd(self, max_iter=100, learn_rate=1.0, delta_thrd=0.1,
                  is_average=True):
        print '-' * 60
        print "START TRAIN SGD:"
        N = len(self.sample_list)    # number of sample
        M = self.feat_size + 1  # number of feature

        # convert to extend sample set
        sample_list = self.sample_list
        # Theta of each class
        omega = self.Theta
        omega_sum = np.zeros(omega.shape)
        rd = 1
        last = 0.
        flag = 0
        start_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print start_clock
        # time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        while rd < max_iter * N:
            if rd % N == 0 and rd != 0:
                loop = rd/N
                error_count, loss = self.__getCost()
                acc = 1 - error_count / float(N)
                # time2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print 'Iter:', loop, '\tCost:', '%.8f' % loss,
                print '\tAcc:', '%.4f' % acc, "\tError: %4d" % error_count
                # print 'start:\t', time1
                # print "stop:\t", time2
                if loop > 1:
                    if last - loss < delta_thrd and last >= loss:
                        print "Reach the minimal loss value threshold!"
                        break
                last = loss
                # time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Choose a random sample
            i = random.randint(0, N - 1)
            result = self.predict2(sample_list[i])
            pred_id = result.index(max(result))
            pred = self.label_set[pred_id]
            label = self.label_list[i]
            label_id = self.label_set.index(label)
            # weight update
            if label != pred:
                for key, val in sample_list[i].items():
                    omega[label_id, key] += learn_rate * val
                    omega[pred_id, key] -= learn_rate * val
            rd += 1
            if is_average:
                omega_sum += omega

        if is_average:
            self.Theta = omega_sum/(rd)
        else:
            self.Theta = omega
        stop_clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "\nTraining process finished."
        print "start time:\t", start_clock
        print "stop time:\t", stop_clock

    def scoreout(self, sample_test):
        # X = self.__getSampleVec(sample_test)
        C = self.class_num
        # pred = self.predict(X)
        pred = self.predict2(sample_test)
        score = {}
        for l in range(C):
            label = self.label_set[l]
            score[label] = pred[l]
        return score

    def probout(self, sample_test):
        score = self.scoreout(sample_test)
        prb = {}
        for key in score.keys():
            prb[key] = self.sigmoid(score[key])
        return prb

    def classify(self, sample_test):
        # X = self.__getSampleVec(sample_test)
        # result = self.predict(X)
        result = self.predict2(sample_test)
        pred_id = result.index(max(result))
        pred = self.label_set[pred_id]
        return pred

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
        self.label_set = list(set(self.label_list))
