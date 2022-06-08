import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
# from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout
import os
from tqdm import tqdm
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,RandomForestClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import sys
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
## check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found.....')
    #sys.exit()
else:
    print('Default GPU device :{}'.format(tf.test.gpu_device_name()))
    
################
def obs(x1,x2):
    F = (2+np.cos(x1)*np.cos(x2))**5
    return np.array(F)  
def generate_init(n):
    x1,x2=[],[]
    for q in range(n):
        x1.append(np.random.uniform(0,10*np.pi))
        x2.append(np.random.uniform(0,10*np.pi))
    return np.array(x1),np.array(x2),np.array([x1,x2]).T


def orcal(x1,x2):
    F = (2+np.cos(x1)*np.cos(x2))**5
    for q,item in enumerate(F):
      if (item >80 and item < 120 ):
        F[q] = 1
      else:
        F[q] =0   
    return np.array(F)  

score = lambda x : np.array([q*(1-q) for q in x])
###################################
class ALS(Model):
    def __init__(self,inshape,drop_rate):
        super(ALS,self).__init__()
       
       
        self.dense1 = Dense(100, input_shape=(inshape,))
        self.dense2 = Dense(100,activation='relu')
        self.dense4 = Dense(1,activation='sigmoid')
        self.dropout= Dropout(drop_rate) 
    
  
    def call(self,input):
 
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense2(x)
        x = self.dense2(x)
        x = self.dense2(x)
        x = self.dense4(x)
        return x
    
model_AL = ALS(2,0.1)
model_AL.compile(optimizer='adam', loss='BinaryCrossentropy')

model_BDT = GradientBoostingClassifier()
######################Active Learning with Random choice of K#############
runs = 200
epoch=1000

badP,goodP = [],[]
x1,x2,X=generate_init(100)
obs1 = orcal(x1, x2)
model_BDT.fit(X, obs1)
for k in tqdm(range(runs),ascii=True,desc='Progress'):  
  X1,X2,x = generate_init(1000)
  ob = orcal(X1, X2)
  Y = ob == 1
  pred = model_BDT.predict(x).flatten()
  sel = round(len(pred[Y])*0.9)
  sel1 = round(len(pred[Y])*0.1)
  pred1 = pred[Y][:sel]
  x_new = x[Y][:sel]
  q = np.argsort(score(pred))[::-1]
  q_new = x[q[:sel1]]
  goodP.append(x_new)
  badP.append(q_new)
  X1 = np.concatenate([x_new,q_new],axis=0)
  ob2 = np.concatenate([ob[Y][:sel],ob[q[:sel1]]],axis=0)
  X= np.append(X,X1,axis=0)
  obs1=np.append(obs1,ob2,axis=0)
  model_BDT.fit(X, obs1) 
 
print('DNN_model- Number of collected points= {}'.format(len(X)))
  
  
  
  
  
  
######
plt.scatter(X[:,0],X[:,1],s=0.5);
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Active Learning- Random batch (points= {:0.0f})'.format(len(X)))
plt.show()
#plt.scatter(np.array(goodP).flatten()[:,0],np.array(goodP).flatten()[:,1])
#plt.scatter(np.array(badP).flatten()[:,0],np.array(badP).flatten()[:,1])





##########AL with Sampling the batch############
def run_AL(runs =20,learning_rate=0.01,n_estimators=100,max_depth=30,print_output=True):
    BDT =GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth)
    x1,x2,X=generate_init(100)
    obs1 = orcal(x1, x2)
    BDT.fit(X, obs1)
    for q in tqdm(range(runs),ascii=True,desc='Progress'): 
      X1,X2,x = generate_init(1000)
      pred = BDT.predict(x).flatten()
      x_new = x[pred==1][:90]
      x_new = np.append(x_new,x[:10],axis=0)
      ob = orcal(x_new[:,0], x_new[:,1])
      Y = ob == 1
      X= np.append(X,x_new[Y],axis=0)
      obs1=np.append(obs1,ob[Y],axis=0)
      BDT.fit(X, obs1) 
    return X
    
XL = run_AL(runs=350)
plt.scatter(XL[:,0],XL[:,1],s=0.5);
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Active Learning- Sampled batch (points= {:0.0f})'.format(len(XL)))
plt.show()

