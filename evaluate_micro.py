# Script for calculating segment-wise (micro-) SDR, SIR and SAR metrics
# Usage: python3 evaluate_micro.py source_dir predictions_dir segment_length
# source_dir should contain the original stem.mp4 files
# predictions_dir should have vocals and accompaniment subfolders [provided relative to the source_dir]

import os
import sys
import numpy as np
import librosa
import stempeg
import statsmodels
from statsmodels import robust
import museval

vocal_SDR = []
vocal_SIR = []
vocal_SAR = []
ctr = 0

source_path = sys.argv[1]
target_path = sys.argv[2]
print('Directory', target_path)

os.chdir(source_path)
dirs = os.listdir()
for file in dirs:
    print(file)
    if (file.endswith('stem.mp4')):
        
       #reference source extraction
        ctr += 1
        ys_stereo,fs = stempeg.read_stems(file,stem_id=0)
        yus = librosa.resample(np.transpose(ys_stereo),fs,22050)
        ys = (yus[0,:]+yus[1,:])/2
        yr_vocals_stereo,fs = stempeg.read_stems(file,stem_id=4)
        yur_vocals = librosa.resample(np.transpose(yr_vocals_stereo),fs,22050)
        yr_vocals = (yur_vocals[0,:]+yur_vocals[1,:])/2
        yr_accomp = ys-yr_vocals
        
        #loading source estimates

        temp = np.load(target_path+'/vocals/'+file[:-9]+'.npz')
        ye_vocals = temp['arr_0'];
        temp = np.load(target_path+'/accompaniment/'+file[:-9]+'.npz')
        ye_accomp = temp['arr_0']

        #calculation of SDR per track

        print("Estimating metrics...")

        ln = len(ye_vocals)
        seglen=int(sys.argv[3])
        t = 0       

        for i in range(0,ln-seglen,seglen):
            if (np.mean(np.abs(yr_vocals[i:i+seglen])) > t) and (np.mean(np.abs(yr_accomp[i:i+seglen])) > t) and (np.mean(np.abs(ye_vocals[i:i+seglen])) > t) and (np.mean(np.abs(ye_accomp[i:i+seglen])) > t):
                references = np.concatenate((np.reshape(yr_vocals[i:i+seglen],(1,seglen)),np.reshape(yr_accomp[i:i+seglen],(1,seglen))),axis=0)
                estimates = np.concatenate((np.reshape(ye_vocals[i:i+seglen],(1,seglen)),np.reshape(ye_accomp[i:i+seglen],(1,seglen))),axis=0)
                [SDR,_,SIR,SAR] = museval.evaluate(references,estimates) #sdr, isr, sir, sar
                vocal_SDR.append(SDR[0])
                vocal_SIR.append(SIR[0])
                vocal_SAR.append(SAR[0])

        print("Current vocal SDR median/mad/mean/std",np.median(np.asarray(vocal_SDR)),robust.mad(np.asarray(vocal_SDR)),np.mean(np.asarray(vocal_SDR)),np.std(np.asarray(vocal_SDR)))
        print("Current vocal SIR median/mad/mean/std",np.median(np.asarray(vocal_SIR)),robust.mad(np.asarray(vocal_SIR)),np.mean(np.asarray(vocal_SIR)),np.std(np.asarray(vocal_SIR)))
        print("Current vocal SAR median/mad/mean/std",np.median(np.asarray(vocal_SAR)),robust.mad(np.asarray(vocal_SAR)),np.mean(np.asarray(vocal_SAR)),np.std(np.asarray(vocal_SAR)))
    
