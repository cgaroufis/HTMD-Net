# Script that
# a) counts the number of segments (of given length) that can be created from an audio dataset
# b) isolates said segments, and groups them (randomly mixed) in "mega-batch" .npz-files

# Usage: python3 preprocess.py source_dir target_dir window_length batch_num
# target_dir is relative to source_dir

import os
import sys
import gc
import numpy as np
import pandas as pd #or not
import librosa
import stempeg
import IPython

source_path = sys.argv[1]
target_path = sys.argv[2]

winlength = int(sys.argv[3])
batch_num = int(sys.argv[4])
hopsize = winlength

os.chdir(source_path)
dirs = os.listdir()
os.mkdir('./'+target_path)
cnt = 0

print("Start counting segments of length ", winlength)
for file in dirs:
    if file.endswith('.mp4'):
        for t in range(0,660,30):
            y,fs = stempeg.read_stems(file,stem_id=0,start=t,duration=30) #due to memory limitations...
            if (len(y) == 0): 
                break
            y = librosa.resample(np.transpose(y),fs,22050)
            yc = (y[0,:] + y[1,:])/2
            for i in range(0,len(yc)-winlength,hopsize):
                cnt = cnt+1
        print("Currently counted", cnt, "segments")

    
datasize = cnt
filesize = datasize//batch_num

order = np.random.permutation((datasize)) 

print("Starting creating", batch_num, "mega-batches from the original data")
for k in range(0,batch_num): 

    audio_chunks = np.zeros((filesize,16384))
    vocal_chunks = np.zeros((filesize,16384))
    
    cnt = 0
    icnt = 0
    cntv = 0
    icntv = 0

    for file in dirs:
        if file.endswith('.stem.mp4'):
            for t in range(0,660,30):
                y,fs = stempeg.read_stems(file,stem_id=0,start=t,duration=30)
                if (len(y) == 0):
                    break
                yc = librosa.resample(np.transpose(y),fs,22050)
                yr = (yc[0,:] + yc[1,:])/2
                for i in range(0,len(yr)-winlength,hopsize):
                    if ((order[cnt] // filesize) == k):
                        audio_chunks[icnt,:] = yr[i:i+hopsize]
                        icnt = icnt+1
                    cnt = cnt+1
        
            gc.collect()
            
            for t in range(0,660,30):
                y,fs = stempeg.read_stems(file,stem_id=4,start=t,duration=30)
                if (len(y) == 0):
                    break
                yc = librosa.resample(np.transpose(y),fs,22050)
                yr = (yc[0,:] + yc[1,:])/2
                for i in range(0,len(yr)-winlength,hopsize):
                    if ((order[cntv]//filesize) == k):
                        vocal_chunks[icntv,:] = yr[i:i+hopsize]
                        icntv = icntv+1
                    
                    cntv = cntv+1

    print("Batch", k+1, "complete")
    filepath = './'+target_path+'/audio_'+str(k)+'.npz'
    np.savez(filepath,audio_chunks)
    filepath = './'+target_path+'/vocals_'+str(k)+'.npz'
    np.savez(filepath,vocal_chunks)
