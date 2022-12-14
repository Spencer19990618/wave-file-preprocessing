import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from librosa import util
import os
# function output the audio with reverb

def remove_ds_store(lst):
    """remove mac specific file if present"""
    if '.DS_Store' in lst:
        lst.remove('.DS_Store')
    return lst

# Assume the room size to be 4x6x3
rt60_tgt = 0.3  # seconds
room_dim = [4, 6, 3]  # meters

audio_dir  = './digit_audio'
file_list = os.listdir(audio_dir)
file_list = remove_ds_store(file_list)
# load the audio
for file in file_list:
    audio_path = os.path.join(audio_dir, file)
    fs, audio = wavfile.read(audio_path)

    print(f"sampling frequency: {fs}")
    print(f"the shape of audio{audio.shape}")


    # invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    # place the source in the room
    room.add_source([2, 2, 1], signal=audio)

    # set the microphones 's location
    mic_locs = np.c_[
        [1.94, 5, 1], [1.98, 5, 1], [2.02, 5, 1], [2.06, 5, 1]
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # run the simulation
    room.simulate()

    out_path  = os.path.join('./reverb_wav', file)
    # save the outcome
    room.mic_array.to_wav(
        out_path,
        norm=True,
        bitdepth=np.int16,
    )
