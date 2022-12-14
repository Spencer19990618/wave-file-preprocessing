from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat, DelaySum
from speechbrain.dataio.dataio import read_audio
from torch.nn.functional import normalize
from torch import mean, max, min
import torchaudio
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input_dir", type=str ,help="input the directory of audio for beamforming ")
parser.add_argument('-o', "--output_dir", type=str ,help="input the directory to save outcome")
args = parser.parse_args()

# audio_dir = './Beamforming-for-speech-enhancement/wav_10-11換722收音/3-前方.wav'

def main():

    def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')
        return lst

    fs = 16000 # Hz
    # microphone array geometry(cm)
    mics = torch.zeros((4,3), dtype=torch.float)
    mics[0,:] = torch.FloatTensor([-6, +0.00, +0.00]) # ch3
    mics[1,:] = torch.FloatTensor([-2, +0.00, +0.00]) # ch4
    mics[2,:] = torch.FloatTensor([2, +0.00, +0.00]) # ch1 
    mics[3,:] = torch.FloatTensor([6, +0.00, +0.00]) # ch2
    # recall the functional classes
    stft = STFT(sample_rate=fs)
    cov = Covariance()
    gccphat = GccPhat()
    delaysum = DelaySum()
    istft = ISTFT(sample_rate=fs)
    audio_dir = args.input_dir
    output_dir = args.output_dir
    audio_list = os.listdir(audio_dir)
    audio_list = remove_ds_store(audio_list)
    # load the audio
    for audio in audio_list:
        audio_path = os.path.join(audio_dir, audio)
        speech = read_audio(audio_path) 
        speech = speech.unsqueeze(0)

        # compute STFT, covariance matrix, GCC-PHAT
        Xs = stft(speech)
        XXs = cov(Xs)
        doas = gccphat(XXs)

        Ys_ds = delaysum(Xs, doas)
        ys_ds = istft(Ys_ds)
        ys_ds = ys_ds.reshape([ys_ds.shape[1],ys_ds.shape[2]])
        ys_ds = ys_ds.mT
        ys_ds_norm = (ys_ds-mean(ys_ds))/(max(ys_ds)-min(ys_ds))
        output_path = os.path.join(output_dir, audio)
        torchaudio.save(output_path, ys_ds_norm, fs)

if __name__=='__main__':
    main()