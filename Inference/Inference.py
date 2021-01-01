#!/usr/bin/env python
# coding: utf-8

# In[3]:


import run as R
from scipy.io import wavfile
# 절대경로
fs, feat  = wavfile.read(r'/home/ubuntu/server/before.wav')
s = R.run(feat)
# 절대경로
wavfile.write("/home/ubuntu/server/after.wav", 16000, s)
print("finished")

# import IPython.display as ipd
# ipd.Audio('output.wav')

