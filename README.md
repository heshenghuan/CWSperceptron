#CWSperceptron

###Introduction
> A Chinese Word Segmentation machine based on perceptron algorithm.<br>
> In order to solve the multiclass classification problem, I used One-vs-All method
> to implemented a multiclass perceptron.<br>
> Using Viterbi algorithm to find the best tag-sequence.

##File specification
* MultiPerceptron.py<br>
  The multiclass perceptron using one-vs-all method.
  
* Dict.py<br>
  A script used to get dictionary information. 
  
* CWSPv4.py<br>
  CWSperceptron with 10+punc features.
  
* CWSPv5.py<br>
  CWSperceptron with 10+punc+dict+type features.

#####note: 
10: 10 base features + punctution information feature<br>
punc: punctutaion information feature<br>
dict: dictionary information features<br>
type: the type features of characters<br>



