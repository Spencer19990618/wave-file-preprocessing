import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from pydub import effects 
from librosa import util
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-ch', "--channels", type=int ,help="input the numver of audio", default=4)
parser.add_argument('-fs', "--sampling_frequency", type=int ,help="input the sampling frequency", default=16000)
parser.add_argument('-i', "--input_dir", type=str ,help="input the directory to precess audio")
parser.add_argument('-o', "--output_dir", type=str ,help="input the directory to save outcome")
parser.add_argument('-m', "--mode", type=int ,help="input the wpe mode", default=1)
args = parser.parse_args()

# function: dereverb the audio 
def main():
    def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')
        return lst

    stft_options = dict(size=512, shift=128)
    channels = args.channels
    sampling_rate = args.sampling_frequency
    delay = 3
    alpha = 0.9999
    taps = 10
    frequency_bins = stft_options['size'] // 2 + 1

    file_dir = args.input_dir
    out_dir = args.output_dir
    file_list = os.listdir(file_dir)
    file_list = remove_ds_store(file_list)

    for file in file_list:

        file_path = os.path.join(file_dir, file)
        fs, audio = wavfile.read(file_path)
        if channels == 1:
            audio = np.expand_dims(audio, axis=1)
            
        print(f"sampling rate: {fs}")
        audio_STFT = stft(audio.T, **stft_options).transpose(1, 2, 0)
        T, _, _ = audio_STFT.shape # shape of STFT: Shape: (frames, frequency bins, channels)
        print(f"The shape of STFT: {audio_STFT.shape}")

        def aquire_framebuffer():
                buffer = list(audio_STFT[:taps+delay, :, :])
                for t in range(taps+delay+1, T):
                    buffer.append(audio_STFT[t, :, :])
                    yield np.array(buffer)
                    buffer.pop(0)
        
        Z_list = []
        if args.mode == 1: # Non-iterative frame online approach
            Q = np.stack([np.identity(channels * taps) for a in range(frequency_bins)])
            G = np.zeros((frequency_bins, channels * taps, channels))

            for Y_step in tqdm(aquire_framebuffer()):
                Z, Q, G = online_wpe_step(
                    Y_step,
                    get_power_online(Y_step.transpose(1, 2, 0)),
                    Q,
                    G,
                    alpha=alpha,
                    taps=taps,
                    delay=delay
                )
                Z_list.append(Z)

            Z_stacked = np.stack(Z_list)
            z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])
        elif args.mode == 2: # Frame online WPE in class fashion
            online_wpe = OnlineWPE(
                taps=taps,
                delay=delay,
                alpha=alpha,
                channel=4,
                frequency_bins=257
            )
            for Y_step in tqdm(aquire_framebuffer()):
                Z_list.append(online_wpe.step_frame(Y_step))

            Z = np.stack(Z_list)
            z = istft(np.asarray(Z).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])
        z = z.T
        print(z.shape)
        z = util.normalize(z)
        out_path = os.path.join(out_dir, file)
        wavfile.write(out_path, fs, z)

if __name__ =='__main__':
    main()
