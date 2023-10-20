import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import random

seed_value=123456

def set_seeds(seed=seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.compat.v1.config.threading.set_inter_op_parallelism_threads(1)
    tf.compat.v1.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=seed_value)
# import os 
# os.environ['PYTHONHASHSEED']=str(seed_value)
# import random
# random.seed(seed_value)
# import numpy as np
# np.random.seed(seed_value)
# import tensorflow as tf
# tf.random.set_seed(seed_value)
# from tensorflow.python.keras import backend as K
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(config=session_conf)
# K.set_session(sess)

#sess = tf.compat.v1.Session(config=config) 
#K.set_session(sess)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import save_model,load_model,Model
from tensorflow.keras.optimizers import Adam, RMSprop, Adamax
from tensorflow.keras.layers import Conv2D, UpSampling2D, AveragePooling2D, MaxPooling2D, Dense,Input, Dropout
from tensorflow.keras.layers import LeakyReLU,Reshape,BatchNormalization, Flatten
from tensorflow.keras.models import save_model
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from IPython import display
import time
from sklearn.mixture import GaussianMixture
import umap
from copy import deepcopy
from PIL import Image
from sklearn.cluster import KMeans
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils
from array import array 
import json
import sys  
sys.path.append(os.path.join('..', 'preprocess_scripts'))
sys.path.append(os.path.join('..', 'utils'))
from generate_features_domain_adaptation import *
#parameters for MLP tuning in source domain
from util_scripts import *
from tqdm import tqdm 


def encoderNN(zdim, imgX):

    x = Dense(zdim, activation='relu', use_bias=True)(imgX)
    encoderX = Model(imgX,x)
    
    return encoderX

def classifierNN(zdim, nofclasses):

    yin =  Input(shape=( zdim,) )
    probX = Dense(units=nofclasses, activation='softmax', use_bias=True)(yin)
    classifier=Model(inputs=[yin],outputs=[probX])
    return classifier

epochs = 15000 #previous value 8000 # change to 20000
epochstep = epochs/100
nofclasses=2
batchsize=200 
nofprojections = 200
#zdim=64
zdim_list=[32,128,256,512]
lamda2_adapt_list=[0.02,0.2,2,20,200]
gamma1_adapt_list=[0.01,0.1,1,10,100]

source="kitchen_&_housewares"
target="electronics"

#change this to avalon locations 
# base_folder="/data/Multi_Domain_Data/BERT_features_v2/"
# source_csv_file="/data/Multi_Domain_Data/parsed_csv_data/kitchen_&_housewares/kitchen_&_housewares_review_splits_combined.csv"
# target_csv_file="/data/Multi_Domain_Data/parsed_csv_data/electronics/electronics_review_splits_combined.csv"
base_folder="/proj/digbose92/domain_adapt_data/BERT_features_v2"
source_csv_file="/proj/digbose92/domain_adapt_data/parsed_csv_data/kitchen_&_housewares/kitchen_&_housewares_review_splits_combined.csv"
target_csv_file="/proj/digbose92/domain_adapt_data/parsed_csv_data/electronics/electronics_review_splits_combined.csv"
layer_name="cls_token"
dataX_train,labelX_train,dataX_test,labelX_test,dataY_train,labelY_train,dataY_test,labelY_test=extract_source_target_feature_set(source,target,base_folder,source_csv_file,target_csv_file,layer_name=layer_name)
# print(dataX_train[0:10,:])
# print(dataY_train[0:10,:])

#dict with keys: {'max_acc_adaptation': ,'acc_before_adaptation':, 'loss_history':, 'acc_history_adaptation':}
run_history_base_folder="/proj/digbose92/domain_adapt_data/BERT_adaptation_features_run_v2"
run_history_folder=os.path.join(run_history_base_folder,source + "_" + target)
if (os.path.exists(run_history_folder) is False):
    os.mkdir(run_history_folder)


imgX = Input(shape=(768,), name="input_img", dtype='float32')  # adapt this if using `channels_first` image data format
labelX=K.placeholder(shape=(None,nofclasses),dtype='float32') #labels of input images oneHot
gamma1_before_adaptation=1e0
gamma2_before_adaptation  = 3e1
lamda2_adapt=2e1
gamma1_adapt=1e1
opt_before_adaptation = Adam(lr=1e-3, decay=1e-5, epsilon=1e-2 , amsgrad=True )
epochs2 = 30000 #change to 20000
epochstep=30
run_number=0
input_dim=768
# for i, zdim in enumerate(zdim_list):
#     dict_temp={}
    #encoderX = encoderNN(zdim=zdim,imgX = imgX)
classifier = classifierNN(input_dim,nofclasses)
loss=[]
    
    
discriminationLoss=K.mean(K.binary_crossentropy(labelX,classifier(imgX)))
regLoss = K.mean(K.square(classifier.weights[0]))
params=classifier.weights 

   
myLoss = discriminationLoss  + gamma1_before_adaptation * regLoss  
       #change lr to 1e-3

updates = opt_before_adaptation.get_updates(myLoss,params)
train = K.function([imgX,labelX],[discriminationLoss],updates)

for itr in tqdm(range(epochs)):
    indTrainDataX,trainLabelX=batchGenerator(labelX_train,batchsize,nofclasses=nofclasses)
    trainDataX=dataX_train[indTrainDataX,...]
    loss.append(train([trainDataX, trainLabelX ]))

perd_label_Y = classifier.predict(dataY_test)
acc_label_Y=(100*float(sum(1*(np.argmax(perd_label_Y,axis=1)==np.argmax(np.squeeze(labelY_test),axis=1))))/perd_label_Y.shape[0])
#print(type(acc_label_Y))
print('Target accuracy before adaptation:%f'%(acc_label_Y))

#dict_temp['acc_before_adaptation']=acc_label_Y

print('Fitting GMMS:')

gmmX = dataX_train
gmmY = np.argmax(labelX_train,axis=1)
yper = classifier.predict(dataX_train)
yperPro = np.max(yper, axis=1)
yper = np.argmax(yper, axis=1)
gmmX = gmmX[yperPro>.98,:]
gmmY = gmmY[yperPro>.98]

#increase max iter to 500
gmmModel =  GaussianMixture(n_components=nofclasses,covariance_type='full', max_iter=500,init_params='kmeans', tol=1e-04) 
gmmModel.fit(gmmX,gmmY)
gmmModelSingle =  GaussianMixture(n_components=1,covariance_type='full') 

for i in range(nofclasses):
        a= gmmX[gmmY==i,:]
        gmmModelSingle.fit(a)
        gmmModel.weights_[i] = gmmY[gmmY==i].shape[0]/gmmY.shape[0]
        gmmModel.covariances_[i] = gmmModelSingle.covariances_[0]
        gmmModel.means_[i] = gmmModelSingle.means_[0]

        gmmModel.precisions_cholesky_[i] = gmmModelSingle.precisions_cholesky_[0]
        gmmModel.precisions_[i] = gmmModelSingle.precisions_[0]

print('Gaussian mixture fitting done')

##### adaptation section #####

labelZ_train = np.concatenate([labelY_train,labelY_test],axis=0)
dataZ_train =  np.concatenate([dataY_train,dataY_test],axis=0)
#encoderXsource = encoderNN(zdim=zdim,imgX=imgX)
#encoderXsource.set_weights(encoderX.get_weights()) 
classifiersource = classifierNN(input_dim,nofclasses)
classifiersource.set_weights(classifier.get_weights()) 

#print('Beginning adaptation')

imgY = Input(shape=(768,), name="input_img")  # adapt this if using `channels_first` image data format

labelY=K.placeholder(shape=(None,nofclasses),dtype='float32') #labels of input images oneHot
#lamda2=1e-2

theta=tf.keras.backend.placeholder(shape = (nofprojections, input_dim), dtype='float32')

thres = .95
loss1 = []

imgY = Input(shape=(input_dim,) )  # adapt this if using `channels_first` image data format
labelY=K.placeholder(shape=(None,nofclasses) ,dtype='float32') #labels of input images oneHot
    #lamda2=2e2
    #lamda2=1e2
    

theta=tf.keras.backend.placeholder(shape = (nofprojections, input_dim), dtype='float32')
labelW = K.placeholder(shape=(None,nofclasses) ,dtype='float32') #labels of input images oneHot
imgW = Input(shape=(768,), name="input_img", dtype='float32')  # adapt this if using `channels_first` image data format
discriminationLoss= K.mean(K.binary_crossentropy(labelY,classifier(imgY)))   \
                        + K.mean(K.categorical_crossentropy(labelW,classifier(imgW)))

#gamma1 = 1e0
    
#change 0.1 to 0.01
matchingLoss=sWasserstein(imgX,imgY,theta,nclass=nofclasses,Cp=None,Cq=None,)
regLoss = K.mean(K.square(classifier.weights[0]))
myLoss= discriminationLoss +lamda2_adapt*matchingLoss + gamma1_adapt * regLoss # + gamma2 * reconLoss

params=classifier.weights #+ decoderX.weights#opt=Adamax(learning_rate=1e-5)
opt=Adam(learning_rate=1e-3)
#opt = Adam(lr=1e-5, decay=1e-6, epsilon=1e-2 , amsgrad=True  ) # Adamax(lr=1e-5)#  very important 
#change lr to 1e-3
updates = opt.get_updates(myLoss,params)
train = K.function(inputs=[imgX,imgY,labelY,theta,imgW,labelW],outputs=[myLoss],updates=updates)
loss2 = []
epoch_pred_Y=[]

#epochstep = epochs2/1000

for itr in tqdm(range(epochs2)):
        indTrainDataY,trainLabelY=batchGenerator(labelZ_train,batchsize,nofclasses=nofclasses)
        trainDataY=dataZ_train[indTrainDataY,...]

        indTrainDataX,trainLabelX=batchGenerator(labelX_train,batchsize,nofclasses=nofclasses)
        trainDataX=dataX_train[indTrainDataX,...]    

        Yembed,Yembedlabel1  = gmmModel.sample(n_samples=10*batchsize)    
        Yembedlabel = utils.to_categorical(Yembedlabel1)   

        perdLabbatchY3 = classifier.predict(Yembed)
        problab = np.max(perdLabbatchY3,axis=1)

        NNN =  1*(problab>thres).sum()    
        trainDataYtemp = Yembed
        Yembed = np.zeros([NNN,input_dim])
        trainLabelYY = np.zeros([NNN,nofclasses])
        count = 0
        for ijk in range(len(problab)):
            if problab[ijk] > thres:
                trainLabelYY[count,:] = 1*(perdLabbatchY3[ijk,:]>thres) 
                Yembed[count,:] = trainDataYtemp[ijk,:]
                count = count + 1    
        Yembedlabel = trainLabelYY
        theta_=generateTheta(nofprojections,input_dim)
        loss2.append(train(inputs=[trainDataY,Yembed,Yembedlabel,theta_ ,trainDataX, trainLabelX ]))

#if(itr % epochstep ==0):
perd_label_Y = classifier.predict(dataY_test)
acc=(100*float(sum(1*(np.argmax(perd_label_Y,axis=1)==np.argmax(np.squeeze(labelY_test),axis=1))))/perd_label_Y.shape[0])
epoch_pred_Y.append(acc)

            #print('Epoch step: %d, Accuracy:%f' %(itr,acc))


print('Ending adaptation')
print('Maximum accuracy after adaptation:%f' %(max(epoch_pred_Y)))

# dict_temp['max_acc_adaptation']=max(epoch_pred_Y)
# dict_temp['loss_history']=loss2
#     dict_temp['acc_adaptation_history']=epoch_pred_Y
#     dict_temp['zdim']=zdim
#     dict_temp['gamma1_before_adaptation']=gamma1_before_adaptation
#     dict_temp['gamma2_before_adaptation']=gamma2_before_adaptation
#     dict_temp['lamda2_adaption']=lamda2_adapt
#     dict_temp['gamma1_adaption']=gamma1_adapt
    
    


    # run_number=run_number+1
    # with open(os.path.join(run_history_folder,'run_'+str(run_number)+".pkl"),"wb") as f:
    #     pickle.dump(dict_temp,f)
