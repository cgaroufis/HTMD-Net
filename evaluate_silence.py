# Script for calculating segment-wise PES (dB), VAD (%) metrics
# Usage: python3 evaluate_silence.py source_dir predictions_dir segment_length
# source_dir should contain the original stem.mp4 files
# predictions_dir should have vocals and accompaniment subfolders [provided relative to the source_dir]
# segment_length corresponds to the segment length for PES (dB) calculation.

import os
import sys
import numpy as np
import librosa
import pyvad
import sklearn
import stempeg
import statsmodels
from statsmodels import robust
import museval

source_path = sys.argv[1]
target_path = sys.argv[2]

print('Directory', target_path)

vocal_PES = []

correct_labels = 0
total_labels = 0

os.chdir(source_path)
dirs = os.listdir()
for file in dirs:
    print("Song name:", file)
    if (file.endswith('.mp4')):
        
        #reference source extraction

        ys_stereo,fs = stempeg.read_stems(file,stem_id=0)
        yus = librosa.resample(np.transpose(ys_stereo),fs,22050)
        ys = (yus[0,:]+yus[1,:])/2
        yr_vocals_stereo,fs = stempeg.read_stems(file,stem_id=4)
        yur_vocals = librosa.resample(np.transpose(yr_vocals_stereo),fs,22050)
        yr_vocals = (yur_vocals[0,:]+yur_vocals[1,:])/2
        yr_accomp = ys-yr_vocals
        
        #loading source estimates

        temp = np.load(target_path+'/vocals/'+file[:-9]+'.npz')
        ye_vocals = temp['arr_0']
        temp = np.load(target_path+'/accompaniment/'+file[:-9]+'.npz')
        ye_accomp = temp['arr_0']

        #calculation of PES per track (input/4 segments)

        print("Estimating predicted energy at silence...")

        ln = len(ye_vocals)
        seglen = int(sys.argv[3])

        for i in range(0,ln-seglen,seglen):
            
            y = librosa.resample(yr_vocals[i:i+seglen],22050,16000)
            vact_labels = pyvad.vad(y,16000,hop_length=20,vad_mode=3) 

            vact_labels = vact_labels[0:-1:320] #vocal activity labels of the reference track

            ye = librosa.resample(ye_vocals[i:i+seglen],22050,16000)
            veact_labels = pyvad.vad(ye,16000,hop_length=20,vad_mode=3)
            
            veact_labels = veact_labels[0:-1:320] #vocal activity labels of the estimated track

            l = len(vact_labels)
            vact_temp  = np.mean(vact_labels) > 0.5 #returns a mean vact_label of the reference segment
            if (vact_temp == 0):
                frame_energy = np.sqrt(np.mean(np.square(ye_vocals[i:i+seglen])))
                vocal_PES.append(max(np.log10(frame_energy),-100))

            correct_labels += np.sum(np.equal(vact_labels,veact_labels))
            total_labels +=l
            
        print("Median/mean predicted energy at silence (dB)", 20*np.median(np.asarray(vocal_PES)), 20*np.mean(np.asarray(vocal_PES)))
        print("Percentage of correctly estimated vocal activity", correct_labels/total_labels)
