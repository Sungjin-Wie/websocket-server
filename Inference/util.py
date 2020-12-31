#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile
import decimal
import os
import pickle

n_concat = 7
n_window = 512
n_overlap = 256
fs = 16000

def calc_sp(audio, mode):
    """Calculate spectrogram. 
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    
    ham_win = np.hamming(n_window)
    
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x

def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments.
    원래는 [시간 길이, 257(FREQ)]으로 되어 있는데,
    이를 [시간길이, 7, 257(FREQ)]으로 만들어 줍니다.
    agg_num = 7 
    hop = 3
    즉 1만큼씩 건너뛰며, 앞뒤 3개씩 자기를 포함해 7개씩 각각 위치에 새로운 값을 만드는 것입니다.
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)

def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    위의 mat_2d_to_3d를 위해 앞뒤에 3개씩 이어붙이는 과정입니다.
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)

def log_sp(x):
    '''spectrogram에 log를 취하는 것입니다.'''
    return np.log(x + 1e-08)




def real_to_complex(pd_abs_x, gt_x):
    """Recover pred spectrogram's phase from ground truth's phase. 
        맨처음 원음의 phase로 복원된 spectrogram의 phase로 대체하여 복원합니다.
        원래 예측된 pred는 로그파워 스펙트로그램입니다.(매그니튜드 순수)
        따라서 매그니튜드와 phase를 합쳐줘야 온전한 음성이 됩니다.
        
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      gt_x: 2d complex array, (n_time, n_freq)
      
    Returns:
      2d complex array, (n_time, n_freq)
    """
    theta = np.angle(gt_x)
    cmplx = pd_abs_x * np.exp(1j * theta)
    return cmplx
    
def half_to_whole(x):
    """Recover whole spectrogram from half spectrogram.
    257의 freq정보를 전체 freq 512로 복구합니다.
    """
    return np.concatenate((x, np.fliplr(np.conj(x[:, 1:-1]))), axis=1)

def ifft_to_wav(x):
    """Recover wav from whole spectrogram
        512의 freq정보를 가지고 inverse 합니다.
    """
    return np.real(np.fft.ifft(x))


def recover_gt_wav(x, n_overlap, winfunc, wav_len=None):
    """Recover ground truth wav. 
    """
    x = half_to_whole(x)
    frames = ifft_to_wav(x)
    (n_frames, n_window) = frames.shape
    s = deframesig(frames=frames, siglen=0, frame_len=n_window, 
                   frame_step=n_window-n_overlap, winfunc=winfunc)
    if wav_len:
        s = pad_or_trunc(s, wav_len)
    return s

def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:np.ones((x,))):    
    """
    Does overlap-add procedure to undo the action of framesig.

    Ref: From https://github.com/jameslyons/python_speech_features
    :param frames: the array of frames.
    -ifft한 후의 음성스펙트로그램.
    (시간, 512)
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    -0
    :param frame_len: length of each frame measured in samples.
    -512
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    -256
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    -[1111111111111111111111111111111111111.............1111111] 같은 매트릭스입니다.
        
    정리하자면, [시간,512]의 것 음원 x을 
    약 시간 곱하기 256 만큼의 길이만큼
    처음의 x[0] => 512의 길이를 가지는 값이겠죠?
    x[0]의 후반 부 반
    이 x[1]의 0~511 의 전반 부 반을 더해서 나눠주는 겁니다.
    이런 작업을 시간 곱하기 256만큼의 길이 만큼 각각 수행해줘서 채워주면서 음원을 완성하는 작업입니다.
    overlap-add작업인데,
    이해가 안되시면
    저한테 알려주시면 알려드리겠습니다.
    글로 어떻게 표현해야할지 모르겠습니다...
    :returns: a 1-D signal.
    """
    
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    
    # assert는 에러메시지 출력을 위한 것입니다.
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    #num_frames = 시간
    #frame_len = 512
    #frame_step = 256    
    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)
    #win = winfunc
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]
    
def round_half_up(number):
    '''반올림 연산입니다.'''
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def pad_or_trunc(s, wav_len):
    """
    길이차이가 생기면 쓰는 함수입니다.
    """
    if len(s) >= wav_len:
        s = s[0 : wav_len]
    else:
        s = np.concatenate((s, np.zeros(wav_len - len(s))))
    return s


def recover_wav(pd_abs_x, gt_x, n_overlap, winfunc, wav_len=None):
    """Recover wave from spectrogram. 
    If you are using scipy.signal.spectrogram, you may need to multipy a scaler
    to the recovered audio after using this function. For example, 
    recover_scaler = np.sqrt((ham_win**2).sum())
    
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      gt_x: 2d complex array, (n_time, n_freq)
      n_overlap: integar. 
      winfunc: func, the analysis window to apply to each frame.
      wav_len: integer. Pad or trunc to wav_len with zero. 
      
      위에 정의된 함수들을 합치는게 전부입니다.
    Returns:
      1d array. 
    """
    x = real_to_complex(pd_abs_x, gt_x)
    x = half_to_whole(x)
    frames = ifft_to_wav(x)
    
    (n_frames, n_window) = frames.shape
    s = deframesig(frames=frames, siglen=0, frame_len=n_window, 
                   frame_step=n_window-n_overlap, winfunc=winfunc)
    if wav_len:
        s = pad_or_trunc(s, wav_len)
    return s



'''
노말라이제이션을 위해 mean값과 std값을 불러오고, 함수를 만들어 줍니다.
'''
def scale(x, mean, std):
    return (x - mean) / std
def inv_scale(x, mean, std,B):
    return (x * std*B + mean)


# In[ ]:




