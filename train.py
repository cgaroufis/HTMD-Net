# Script for defining and training an HTMD-Net model
# supposes that data have been preprocessed using the preprocess.py script
# and are saved in a folder w/ train and valid subfolders
# Usage: python3 train.py source_dir intermediate_loss final_loss intermediate_loss_weight final_loss_weight
# eg: python3 train.py /data/musdb18/ mse 0.5 mse 1

import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense,Bidirectional,LSTM,Lambda,Conv2DTranspose,DepthwiseConv2D,Input,Multiply,Add,BatchNormalization,SeparableConv1D,Convolution1D,AveragePooling1D,UpSampling1D,concatenate,LeakyReLU

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=32, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2),name='masked_source')(x)
    return x

def DepthwiseConv1D(input_tensor, kernel_size, dilation_rate):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = DepthwiseConv2D(kernel_size=(3, 1), padding='same', dilation_rate=(dilation_rate,1))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

#Non-logscaled SI-SDR (calculated over the nonsilent segments)

def si_sdr(y_true,y_pred):
    epsilon = 0.0001
    mask = K.cast(K.mean(K.square(y_true)) > epsilon,K.floatx())
    scale = K.batch_dot(y_true,y_true,axes=(1,1)) + epsilon #nan avoidance
    scale_ = K.batch_dot(y_true,y_pred,axes=(1,1)) 
    scale_ratio = scale_/scale
    true_scaled = scale_ratio*y_true
    num = K.batch_dot(true_scaled,true_scaled,axes=(1,1))
    denom = K.batch_dot(true_scaled - y_pred, true_scaled - y_pred, axes=(1,1)) + 0.0000001
    return -mask*num/denom

path = sys.argv[1]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)    

# Mask-applying (convtasnet-like) component

x = Input(shape=(16384,1))
y = x
layerouts = []

x = Input((16384,1))
r = x
y = x
x = Convolution1D(512,16,strides=8,padding='same')(x)
e = x

#Separator (calculates a mask that be applied in the input representation)

x = BatchNormalization()(x)
x = Convolution1D(128,1,padding='same')(x)

for i in range (0,1): #num of blocks
    for k in range (0,9): #num of dilations/resolutions (increase them to get RF > sample_len for the encoder)

        z = x
        x = Convolution1D(512,1,padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = DepthwiseConv1D(x,3,dilation_rate=2**k)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        t = Convolution1D(128,1,padding='same')(x)
        if (i !=0 ) or (k != 0):
            s = Add()([t,s]) #skip connection, gradually adding the masks from each dilated block
        else:
            s = t
        if (i != 2) or (k != 8):
            x = Convolution1D(128,1,padding='same')(x) #1x1 conv to reshape the feature map.
            x = Add()([z,x]) #output connection, with a residual block to the input

x = LeakyReLU()(s)
x = Convolution1D(512,1,padding='same',activation='sigmoid')(x)

x = Multiply()([e,x]) #multiplies the masks
t = Conv1DTranspose(x,1,16,strides=8,padding='same')

zz = t

#attaching a WaveUNet-like architecture for enhancement of the source/interference removal.
x = t
z = t

for i in range(0,11):
    if (i == 0): 
        t = Convolution1D(12,5,padding='same')(x) #reducing the receptive field of 1st layer as per Cohen-Hadria et al 
    else:
    	t = Convolution1D(12*(i+1),15,padding='same')(x)
    t = LeakyReLU()(t)
    x = AveragePooling1D(pool_size=2)(t)
    layerouts.append(t)

t = Bidirectional(LSTM(168,activation=None,return_sequences=True),merge_mode='ave')(x)
t = Bidirectional(LSTM(168,activation=None,return_sequences=True),merge_mode='ave')(t)
t = LeakyReLU()(t)

for i in range(10,-1,-1):
    u = UpSampling1D(size=2)(t)
    w = concatenate([u,layerouts[i]],axis=2)
    if (i > 0):
        t = Convolution1D(12*(i+1),5,padding='same')(w)
    else:
        t = Convolution1D(12,5,padding='same')(w) 
    t = LeakyReLU()(t)

w = concatenate([t,z],axis=2)
t = Convolution1D(1,5,padding='same',activation='tanh',name='vocal_source')(w)

TestNet = Model(input=y,outputs=[t,zz])

int_loss = sys.argv[2]
fin_loss = sys.argv[3]

int_weight = float(sys.argv[4])
fin_weight = float(sys.argv[5])

losses = {'masked_source':int_loss,'vocal_source':fin_loss}
loss_weights = {'masked_source':int_weight, 'vocal_source': fin_weight}
metrics = {'masked_source': [si_sdr,'mse'], 'vocal_source': [si_sdr,'mse']}
print(TestNet.summary())
opt = Adam(lr = 0.0001)
TestNet.compile(loss=losses,optimizer=opt,loss_weights=loss_weights,metrics=metrics)

minvalLoss = 9999
iters = 150

noImpr_epochs = 0
patience_epochs = 20

train_path = path+'/train'
valid_path = path+'/valid'
tb = len(os.listdir(train_path))//2
vb = len(os.listdir(valid_path))//2

for k in range(0,iters):
    print("Iteration ",k+1)    
    for i in range(0,tb):
        filename = train_path+'/audio_'+str(i)+'.npz'
        temp = np.load(filename)
        input_batch = temp["arr_0"]
        input_batch = np.reshape(input_batch,(input_batch.shape[0],16384,1))
        
        filename = train_path+'/vocal_'+str(i)+'.npz'
        temp = np.load(filename)
        output_batch = temp["arr_0"]
        output_batch = np.reshape(output_batch,(input_batch.shape[0],16384,1))

        history = TestNet.fit(x=input_batch,y=[output_batch,output_batch],epochs=k+1,batch_size=8,verbose=2,validation_data=None,initial_epoch=k)
    
    valLoss = 0
    for i in range(0,vb):
        filename = valid_path+'/audio_'+str(i)+'.npz'
        temp = np.load(filename)
        val_input_batch = temp["arr_0"]
        val_input_batch = np.reshape(val_input_batch,(val_input_batch.shape[0],16384,1))

        filename = valid_path+'/vocal_'+str(i)+'.npz'
        temp = np.load(filename)
        val_output_batch = temp["arr_0"]
        val_output_batch = np.reshape(val_output_batch,(val_input_batch.shape[0],16384,1))

        evals = TestNet.evaluate(x=val_input_batch,y=[val_output_batch,val_output_batch],verbose=0)
        print(evals)
        valLoss = valLoss + evals[0]
        
    overall_valLoss = valLoss/vb
    print("Overall loss", overall_valLoss)
    if (overall_valLoss < minvalLoss):
        minvalLoss = overall_valLoss
        if (k > 0):
            os.remove(best_filename)

        best_filename = 'htmdnet_iter'+str(k+1)+'.h5'
        TestNet.save(best_filename)
        noImpr_epochs = 0
    else:
        noImpr_epochs = noImpr_epochs + 1

    if (noImpr_epochs > patience_epochs):
        print("Training finishing...")
        break

