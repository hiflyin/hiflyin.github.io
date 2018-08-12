---
layout: post
title: Vanilla LSTM python equations and code explained
---

What is the article about?
Implementing and explaining the equations of a plain vanilla RNN in pure python. Explaining the building blocks of an
RNN and potential architectures. The code for this article is on [github here](https://github.com/hiflyin/Vanilla-LSTM/blob/master/rnn.py).
The prediction task example is: learn to predict the sum of two numbers

Why Recursive Neural Networks?
Recursive Neural Networks have become very popular in the latest decade because the have opened the door to embedding 
short memory and attention into deep neural networks. As an intuitive definition: we say that a model has short term
memory if it can distinguish the information it has received from the previous few samples in order to decide what 
to learn about the new sample it sees (as opposed to long term memory which would be for example the coefficients in a
plain  linear regression). We say that a model embeds attention if it has a mechanism of looking and deciding through
a sequence of samples at once rather than sequentially and picking what information to retain (where to pay attention).
These are more advanced architectures and concepts with very active areas of research. However, the building 
block concept is the plain RNN. 

Generic RNN layer in pure python
The generic RNN in pure python is as follows:

| xx | xx    |


This 

RNN network will learn to output the value of one bit in the binary repr of the number given the 2 input bits
of the input numbers at each position.
For example for 2+3 = 5 which in 3-bit binary is 010 + 011 = 101, the network will be trained to add 010 + 011
bit by bit from right to left having memorized the carry bit from the previous position. As such, the first
inputs are 0 and 1 and output 1, then next inputs are 1 and 1 and output is 0 with 1 left to carry and finally,
the last inputs are 0 and 0 which is 0 but there's a 1 from the previous inputs so the final result is 1.

