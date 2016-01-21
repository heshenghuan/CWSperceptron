#CWSperceptron

###Introduction
> A Chinese Word Segmentation machine based on perceptron algorithm.<br>
> In order to solve the multiclass classification problem, I used One-vs-All method
> to implemented a multiclass perceptron.<br>
> Using Viterbi algorithm to find the best tag-sequence.

##File specification
* MultiPerceptron.py<br>
  The multiclass perceptron.
  
* Dict.py<br>
  A script used to get dictionary information. 
  
* CWSPv4.py<br>
  CWSperceptron with 10+punc+dict+type features.<br>
  But only used unigram&bigram list to record feature.
  
* CWSPv6.py<br>
  CWSperceptron with 10+punc+dict+type features.<br>
  But used unigram, bigram and trigram list to record feature.

* resources/*<br>
  resource files include dictionary and some special characters list.

* testCase/*<br>
  test files that used for pretreatment, training and segmentation.<br>
  You can use the python file processing corresponding corpus.

#####notes: 
10: 10 base features + punctution information feature<br>
punc: punctutaion information feature<br>
dict: dictionary information features<br>
type: the type features of characters<br>



