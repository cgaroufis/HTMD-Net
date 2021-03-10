# Script for extracting the vocal [+accompaniment] sources from "mixture" mp4 files using htmd-net.
# Usage: python3 extract_sources.py net_model source_dir target_dir
# target_dir is given relative to the source_dir

import os
import sys
import numpy as np
import librosa
import stempeg
import keras
import keras.losses
from keras.models import load_model
from keras import backend as K

def si_sdr(y_true,y_pred):
    return 1000*K.mean((K.square(y_true - y_pred)))

keras.metrics.si_sdr = si_sdr
os.nice(10)

NetModel = load_model(sys.argv[1])
source_path = sys.argv[2]
target_path = sys.argv[3]

#os.makedirs(target_path,exist_ok=True)
#print("Directory created", target_path)

os.chdir(source_path)
os.makedirs('./'+target_path+'/vocals',exist_ok=True)
os.makedirs('./'+target_path+'/accompaniment',exist_ok=True)

ctr = 0
dirs = os.listdir()
for file in dirs:
    print("Song name:", file)
    if (file.endswith('.mp4')):
        
        ctr = ctr+1
        if ctr < 92:
        
            #reference source extraction

            ys_stereo,fs = stempeg.read_stems(file,stem_id=0)
            yus = librosa.resample(np.transpose(ys_stereo),fs,22050)
            ys = (yus[0,:]+yus[1,:])/2
            yr_vocals_stereo,fs = stempeg.read_stems(file,stem_id=4)
            yur_vocals = librosa.resample(np.transpose(yr_vocals_stereo),fs,22050)
            yr_vocals = (yur_vocals[0,:]+yur_vocals[1,:])/2
            yr_accomp = ys-yr_vocals
        
            #calculation of source estimates

            print("Estimating sources...")

            ct = 0
            ln = len(yr_accomp)

            ye_vocals = np.zeros((ln,))
            ye_accomp = np.zeros((ln,))
            for i in range(0,ln-16384,16384):
            
                ys_segment = ys[i:i+16384]
                [temp,_] = NetModel.predict(np.reshape(ys_segment,(1,16384,1))) #to extract the bottleneck output, use the 2nd return argument
                ye_vocals[i:i+16384] = np.reshape(temp,(16384,))
                ye_accomp[i:i+16384] = ys_segment - ye_vocals[i:i+16384]
        
            np.savez(target_path+'/vocals/'+file[:-9],ye_vocals)
            np.savez(target_path+'/accompaniment/'+file[:-9],ye_accomp)
        
