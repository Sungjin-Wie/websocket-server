#!/usr/bin/env python
# coding: utf-8

# In[3]:


import model_cpu as M
import util as U
import numpy as np
import torch

def recon(audio,model, mean, std,b):
    n_concat = 7
    n_window = 512
    n_overlap = 256
    fs = 16000
    mixed_cmplx_x = U.calc_sp(audio, mode='complex')

    #ABS값으로 변환합니다.
    mixed_x = np.abs(mixed_cmplx_x)
    n_pad = int((n_concat - 1) / 2)
    #앞 뒤로 3개씩 패딩해줍니다.
    mixed_x = U.pad_with_border(mixed_x, n_pad)
    #로그값을 입혀줍니다.
    mixed_x = U.log_sp(mixed_x)
    
        #scaling을 실행합니다.
    mixed_x = U.scale(mixed_x, mean,std)
    #[시간,주파수]->[시간,7,주파수]
    mixed_x_3d = U.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

    #pytorch 모델을 사용하기 위해 numpy배열을 torch tensor로 변환합니다.
    mixed_x_3d = torch.from_numpy(mixed_x_3d )
    mixed_x_3d = mixed_x_3d.view(mixed_x_3d.shape[0],1,7,257)

    #변환된 스펙트로그램을 model에 넣어 결과값을 빼내옵니다.
    pred = model(mixed_x_3d.float())
    #결과값을 원활하게 사용하기 위해 numpy값으로 변환합니다.
    pred = pred.cpu().detach().numpy()

    #결과값을 더욱 좋은 음성으로 복원하기 위한 값입니다.
    B = b
    #inverse scaling을 실행합니다.
    pred = U.inv_scale(pred, mean, std,B)

    #아까 로그를 취해주었듯 exp를 해줍니다.
    pred_sp = np.exp(pred)

    s = U.recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hanning)
    #완벽한 복원을 위해 값을 곱해줍니다.
    
    s *= np.sqrt((np.hamming(n_window)**2).sum())

    #int16타입으로 변환후 쓰고 읽어옵니다.
    s = s.astype(np.int16)
    return s

def run(feat):
    # Load model
    model = M.Load_model()
    mean = np.load('/home/ubuntu/server/Inference/mean15dB.npy')
    std = np.load('/home/ubuntu/server/Inference/std15dB.npy')
    
    B=1.25
    reconstructed = recon(feat, model, mean,std,B)
    
    return reconstructed

