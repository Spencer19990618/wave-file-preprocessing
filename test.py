from helper_func import pcm2wav 
import librosa
import numpy as np
import os

pcm_file_dir = './pcm_record'

if not os.path.exists('wav_folder'):
    os.makedirs('wav_folder')

pcm_file_list = os.listdir(pcm_file_dir)
channel  = 4
sr = 16000

for pcm in pcm_file_list:
    pcm_dir = os.path.join(pcm_file_dir, pcm)
    out_path = './wav_folder/' + pcm.split('.')[0] + '.wav'
    pcm2wav(pcm_dir, out_path, 4, sr)
    myvoice, _ = librosa.load(out_path, mono=False)
    print(myvoice.shape)