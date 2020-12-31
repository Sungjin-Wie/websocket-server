#!/usr/bin/env python
# coding: utf-8

# In[3]:


import run as R
from scipy.io import wavfile

fs, feat  = wavfile.read(r'/Users/sungjin/dev/socketserver/Inference/cafebefore.wav')
s = R.run(feat)
wavfile.write("/Users/sungjin/dev/socketserver/Inference/cafeafter.wav", 16000, s)
print("finished")

import IPython.display as ipd
ipd.Audio('output.wav')

