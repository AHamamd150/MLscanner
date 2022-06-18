import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout
from scan_general import *
import os
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
import warnings
## check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found.....')

else:
    print('Default GPU device :{}'.format(tf.test.gpu_device_name()))
################################
################################


class scan():
    def __init__(self,LL,function_dim,th_value,standard_deviation):
      self.th_value = th_value
      self.standard_deviation = standard_deviation
      self.LL=LL
      self.function_dim = function_dim 


    def run_DNN(self,runs=20,neurons = 100,activation='relu',epoch=1000,batch_size=500,batch_L=1000,batch_K=100,print_output=True):
        dnn = Sequential()
        dnn.add(Dense(neurons, input_shape=(self.function_dim,)))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(1))

        dnn.compile(optimizer='adam', loss='mse')    
        Xf,obsf=run_train(batch_L)

        dnn.fit(Xf, obsf, epochs=epoch, verbose=0)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(runs):
            x = generate_init_HEP(batch_L)
            pred = dnn.predict(x).flatten()
            xsel = x[pred<self.th_value]           
            xsel1 = np.append(xsel[:round(batch_K*0.9)],x[-round(batch_K*0.1):],axis=0)
            _,obs2 = run_loop(xsel1)
            X = np.append(X, xsel1, axis=0)
            obs1 = np.append(obs1, obs2)
            dnn.fit(X, obs1,epochs=epoch,batch_size=batch_size, verbose=0)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        return X,obs1


    def run_GBR(self,runs=20,learning_rate=0.01,n_estimators=100,max_depth=30,batch_L=1000,batch_K=100,print_output=True):
        GBR =GradientBoostingRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth)
        Xf,obsf=run_train(batch_L)
        GBR.fit(Xf, obsf)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(runs):
            x = generate_init_HEP(batch_L)
            pred = GBR.predict(x).flatten()
            xsel = x[pred<self.th_value]           
            xsel1 = np.append(xsel[:round(batch_K*0.9)],x[-round(batch_K*0.1):],axis=0)
            _,obs2 = run_loop(xsel1)
            X = np.append(X, xsel1, axis=0)
            obs1 = np.append(obs1, obs2)
            GBR.fit(X, obs1)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        return X
    
    def run_RFR(self,runs=20,learning_rate=0.01,n_estimators=100,max_depth=30,batch_L=1000,batch_K=100,print_output=True):
        RFR =RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
        Xf,obsf=run_train(batch_L)
        RFR.fit(Xf, obsf)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(runs):
            x = generate_init_HEP(batch_L)
            pred = RFR.predict(x).flatten()
            xsel = x[pred < self.th_value]           
            xsel1 = np.append(xsel[:round(batch_K*0.9)],x[-round(batch_K*0.1):],axis=0)
            _,obs2 = run_loop(xsel1)
            X = np.append(X, xsel1, axis=0)
            obs1 = np.append(obs1, obs2)
            RFR.fit(X, obs1)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        return X,obs1


    def run_SVMRBF(self,runs=20, C=100, gamma=0.1,epsilon=0.1,batch_L=1000,batch_K=100,print_output=True):
        SVMRBF = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        Xf,obsf=run_train(batch_L)
        SVMRBF.fit(Xf, obsf)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(runs):
            x = generate_init_HEP(batch_L)
            pred = SVMRBF.predict(x).flatten()
            xsel = x[pred < self.th_value]           
            xsel1 = np.append(xsel[:round(batch_K*0.9)],x[-round(batch_K*0.1):],axis=0)
            _,obs2 = run_loop(xsel1)
            X = np.append(X, xsel1, axis=0)
            obs1 = np.append(obs1, obs2)
            SVMRBF.fit(X, obs1)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        return X


    def run_SVMPOLY(self,runs =20,degree=3, C=100, gamma=0.1, epsilon=0.1,batch_L=1000,batch_K=100,print_output=True):
        SVMPOLY = SVR(kernel="poly", degree=degree, C=C, gamma=gamma, epsilon=epsilon)
        Xf,obsf=run_train(batch_L)
        SVMPOLY.fit(Xf, obsf)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(runs):
            x = generate_init_HEP(batch_L)
            pred = SVMPOLY.predict(x).flatten()
           xsel = x[pred < self.th_value]           
            xsel1 = np.append(xsel[:round(batch_K*0.9)],x[-round(batch_K*0.1):],axis=0)
            _,obs2 = run_loop(xsel1)
            X = np.append(X, xsel1, axis=0)
            obs1 = np.append(obs1, obs2)
            SVMPOLY.fit(X, obs1)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        return X,obs1  
        
        
        
    def plot_model(self,input,title=''):
        plt.figure(figsize=(4,4))
        plt.scatter(input[:, 0],input[:, 1],s=1)
        plt.xlabel(r'$X_1$',fontsize=10);
        plt.ylabel(r'$X_2$',fontsize=10);
        plt.title(title,fontsize=10)
        c = plt.colorbar(orientation="horizontal");
        c.set_label('likelihood',size=10);



##########################################
# Create an instanace of the calss to access all functions    #
##########################################
model = scan(LL=0.9,function_dim=TotVarScanned,th_value=95,standard_deviation=20)
########## Run specific ML model #################
output_dd,obs_dd =model.run_DNN(runs=100,neurons = 100,activation='relu',batch_K=200,batch_L=100000,print_output=True) #Random Forest regressor
output_rf,obs_rf =model.run_RFR(runs=20,n_estimators=300,max_depth=100,batch_K=200,batch_L=10000,print_output=True) #DNN regressor
output_GB,obs_GB=model.run_GBR(runs=40,n_estimators=100,print_output=False) #Gradient boost regressor
output_RBF,obs_RBF =model.run_SVMRBF(runs=50,print_output=True) #SVM regressor with rbf kernel
output_POLY,obs_POLY =model.run_SVMPOLY(runs =5,degree=2, C=1, gamma=0.1, epsilon=0.1,print_output=True)#SVM regressor with poly kernel
