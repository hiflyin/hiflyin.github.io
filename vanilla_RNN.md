---
layout: post
title: Vanilla RNN python equations explained
---

RNN network will learn to output the value of one bit in the binary repr of the number given the 2 input bits
of the input numbers at each position.
For example for 2+3 = 5 which in 3-bit binary is 010 + 011 = 101, the network will be trained to add 010 + 011
bit by bit from right to left having memorized the carry bit from the previous position. As such, the first
inputs are 0 and 1 and output 1, then next inputs are 1 and 1 and output is 0 with 1 left to carry and finally,
the last inputs are 0 and 0 which is 0 but there's a 1 from the previous inputs so the final result is 1.

```python
import numpy as np

```