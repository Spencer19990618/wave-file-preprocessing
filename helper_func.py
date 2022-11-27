import os
import wave
import librosa
from librosa import util
from scipy.io import wavfile

# Usage: pcm2wav(pcm_path, out_path, channel, sample_rate)
def pcm2wav(pcm_path, out_path, channel, sample_rate):
    with open(pcm_path, 'rb') as pcm_file:
        pcm_data = pcm_file.read()
        pcm_file.close()
    with wave.open(out_path, 'wb') as wav_file:
        wav_file.setparams((channel, 16 // 8, sample_rate, 0, 'NONE', 'NONE'))
        wav_file.writeframes(pcm_data)
        wav_file.close()

def ch4_norm(wav_path, save_path):
    y, sr = librosa.load(wav_path,mono=False)
    y = util.normalize(y.T)
    wavfile.write(save_path, sr, y)
# if __name__ == '__main__':
#     dir = r"C:\Users\Administrator\Desktop\test"
#     out_dir = dir + r"\outwav"
#     sample_rate = 48000
#     channel = 1
#     out_path = os.path.join(out_dir, "16k.wav")
#     pcm2wav(os.path.join(dir, "16k.pcm"), out_path, channel, sample_rate)