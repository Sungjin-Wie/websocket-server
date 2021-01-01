#!/usr/bin/env python
# coding: utf-8

# In[4]:


from moviepy.editor import *
import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy.io import wavfile
import model_cpu as M
import run as R

videoclip = VideoFileClip("sample.mp4")
audioclip = videoclip.audio

videoclip.audio.write_audiofile("audio.wav", fps=16000)
fs, feat  = wavfile.read(r'audio.wav')
Ls= feat[:,0]

model = M.Load_model()
mean = np.load('mean15dB.npy')
std = np.load('std15dB.npy')
reconstructed = R.recon(Ls, model, mean,std,1.25)
wavfile.write("output.wav", 16000, reconstructed)

snd = AudioFileClip("output.wav")
videoclip.audio = snd
videoclip.write_videofile("new video.mp4")

