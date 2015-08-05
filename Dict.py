# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 15:40:18 2015

@author: heshenghuan
"""

import codecs

class Dict(object):
    def __init__(self):
        self.dic = {}
        #self.entry_num = 0
    
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
    
    def readDict(self, dict_file):
        print "Loading dict data and building the dictionary for CWSP",
        input_data = codecs.open(dict_file, 'r', 'utf-8')
        entry_num = 0
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                entry_num += 1
            if entry_num%1000 == 0 and entry_num !=0:
                print '.',
            word = rawText.split()[0]      #remove the space
            length = len(word)
            #print word,length,rawText,len(rawText)
            if length > 1:
                tag = self.getTag([word])
                #print tag
                for i in range(length):
                    if self.dic.has_key(word[i]):
                        info = self.dic[word[i]]
                        if length > info[0]:
                            self.dic[word[i]] = [length, tag[i]]
                    else:
                        self.dic[word[i]] = [length, tag[i]]
            else:
                self.dic[word[0]] = [1, "S"]
        print "\nLoading Corpus done."
        
    def saveDict(self, outfile):
        """Save the dictionary information"""
        output = codecs.open(outfile,'w','utf-8')
        for i in self.dic.keys():
            output.write(i)
            output.write(' ')
            output.write(str(self.dic[i][0]))
            output.write(' ')
            output.write(self.dic[i][1])
            output.write("\n")
        output.close()
    
    def loadDict(self, dictfile):
        """Load the dictionary information from file."""
        input_data = codecs.open(dictfile, 'r', 'utf-8')
        entry_num = 0
        for line in input_data.readlines():
            rawText = line.strip()
            if rawText == '':
                continue
            else:
                entry_num += 1
            word = rawText.split()      #remove the space
            self.dic[word[0]] = [int(word[1]), word[2]]