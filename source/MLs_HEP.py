import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout
from auxiliary import *
import sklearn
import os
import sys
import pickle
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor, RandomForestClassifier
import warnings
from imblearn.over_sampling import SMOTE
#########################
if  not os.path.exists(str(os.getcwd())+'/results/'):
    os.mkdir(str(os.getcwd())+'/results/') 
path = str(os.getcwd())+'/results/'
################################
############ colors ##############
def Red(prt): print("\033[91m {}\033[00m" .format(prt))
def Green(prt): print("\033[92m {}\033[00m" .format(prt))
def Yellow(prt): print("\033[93m {}\033[00m" .format(prt))
#############################################
################################
if not os.path.exists(str(os.getcwd())+'/work'):
    os.mkdir(str(os.getcwd())+'/work')
p=Pool(processes=N_Cores)
p.map(dir_prepare,range(1,N_Cores+1),chunksize=1)   
##################################################

args = str(sys.argv[-1])
l=['DNNR','GBR','RFR','SVR_RBF','SVR_POLY','DNNC','RFC']
if args not in l: sys.exit(Red('The given model name is wrong, please check the available models by running ./run.sh --help '))
class scan():
    def __init__(self,iteration,L1,L,K, function_dim,th_value,period,frac,K_smote):
        self.iteration= iteration
        self.L1 = L1
        self.L =L
        self.K = K
        self.th_value = th_value
        self.function_dim = function_dim 
        self.period = period
        self.frac = frac
        self.K_smote=K_smote

    def labeler(self,x,th):
        ll=[]
        for q,item in enumerate(x):
            if item < th:
                ll.append(1)
            else:
                ll.append(0)
        return np.array(ll).ravel()        

    
    def run_DNNR(self,neurons = 100,activation='relu',epoch=1000,batch_size=1000,print_output=True):
        dnn = Sequential()
        dnn.add(Dense(neurons, input_shape=(self.function_dim,)))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(1))
        dnn.compile(optimizer='adam', loss='mae') 
        A1=[]
        p1=Pool(processes=N_Cores)      
        A1.append(p1.map(run_train_p,range(1,N_Cores+1),chunksize=10))
        print(np.array(A1).shape)
        A1=np.array(A1).reshape(N_Cores,2).T 
        print(np.array(A1).shape)  
        Xf=np.array(flatten(A1[0]))
        obs1f=np.array(flatten(A1[1]))
        dnn.fit(np.array(Xf), np.array(obs1f), epochs=epoch, verbose=1)
        X = Xf[obs1f<self.th_value]
        obs1 = obs1f[obs1f<self.th_value]
        effc,accu = [],[] 
        #for q in range(self.iteration):
        while len(obs1) < 20000 :
            x = generate_init_HEP(self.L)
            pred = dnn.predict(x).flatten()
            xsel = x[pred<self.th_value]           
            xsel1 = np.append(xsel[:round(self.K*(1-self.frac))],x[-round(self.K*self.frac):],axis=0)
            xsel2,obs2 = run_loop1(xsel1)
            eff = len(obs2[obs2<self.th_value])/len(obs2)
            effc.append(eff)
            accu.append(len(obs1))
            print('Effeiciency=  ', eff)
            X = np.append(X, xsel2, axis=0)
            obs1 = np.append(obs1, obs2)
            if len(X) == 0: sys.exit('No accumlated points found for the first run, please try different scan ranges.')
            #if (q%self.period==0 or q+1 == self.iteration):

            dnn.fit(X, obs1,epochs=epoch,batch_size=batch_size, verbose=0)
            #else:
            #    dnn.fit(xsel2, obs2,epochs=epoch,batch_size=batch_size, verbose=0)    
            #if print_output == True:
            #    print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        
        dnn.save(path+'Model_dnn_regression.sav')
        np.savetxt(path+'Accumelated_points_Model_dnn_Regression.txt',X) 
        np.savetxt(path+'Eff_Model_dnn_Regression.txt',effc)
        np.savetxt(path+'Accu_Model_dnn_Regression.txt',accu) 
        np.savetxt(path+'HiggsSignals_chiSquar_Model_dnn_Regression.txt',obs1)    
        Yellow('''\t Model saved in ./results/Model_dnn_Regression.sav
        Accumelated points saved in ./results/Accumelated_points_Model_dnn_Regression.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_dnn_Regression.txt ''')
        return 


    def run_GBR(self,learning_rate=0.01,n_estimators=100,max_depth=30,print_output=True):
        GBR =GradientBoostingRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth)
        A=[]
        p1=Pool(processes=N_Cores)      
        A.append(p1.map(run_train_p,range(1,N_Cores+1),chunksize=1)   )
        A=np.array(A).reshape(N_Cores,2).T   
        Xf=np.array(flatten(A[0]))
        obsf=np.array(flatten(A[1]))
        GBR.fit(Xf, obsf)
        X = Xf[obsf<self.th_value]
        obs1 = obsf[obsf< self.th_value]
        for q in range(self.iteration):
            x = generate_init_HEP(self.L)
            pred = GBR.predict(x).flatten()
            xsel = x[pred<self.th_value]           
            xsel1 = np.append(xsel[:round(self.K*(1-self.frac))],x[-round(self.K*self.frac):],axis=0)
            xsel2,obs2 = run_loop1(xsel1)
            X = np.append(X, xsel2, axis=0)
            obs1 = np.append(obs1, obs2)
            if len(X) == 0: sys.exit('No accumlated points found for the first run, please try different scan ranges.')
            if (q%self.period==0 or q+1 == self.iteration):

                GBR.fit(X, obs1)
            else:
                GBR.fit(xsel2, obs2)    
            if print_output == True:
                
                print('GradientBoostingRegressor model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        f=(path+'Model_GradientBoostingRegressor.sav')
        pickle.dump(GBR, open(f, 'wb'))    
        np.savetxt(path+'Accumelated_points_Model_GradientBoostingRegressor.txt',X) 
        np.savetxt(path+'HiggsSignals_chiSquar_Model_GradientBoostingRegressor.txt',obs1)    
        Yellow('''\t Model saved in ./results/Model_GradientBoostingRegressor.sav
        Accumelated points saved in ./results/Accumelated_points_Model_GradientBoostingRegressor.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_GradientBoostingRegressor.txt ''')    
                
        return 
                         
    def run_RFR(self,learning_rate=0.01,n_estimators=300,max_depth=50,print_output=True):
        RFR =RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
        Xf,obs1f=run_train(self.L1)
        RFR.fit(Xf, obs1f)
        X   = np.empty(shape=[0,self.function_dim]) 
        obs1= np.empty(shape=[0,1])
        for q in range(self.iteration):
            x = generate_init_HEP(self.L)
            pred = RFR.predict(x).flatten()
            xsel = x[pred < self.th_value]           
            xsel1 = np.append(xsel[:round(self.K*(1-self.frac))],x[-round(self.K*self.frac):],axis=0)
            xsel2,obs2 = run_loop(xsel1)
            X = np.append(X, xsel2, axis=0)
            obs1 = np.append(obs1, obs2)
            if len(X) == 0: sys.exit('No accumlated points found for the first run, please try different scan ranges.')
            if (q%self.period==0 or q+1 == self.iteration):
                RFR.fit(X, obs1)
            else:
                RFR.fit(xsel2,obs2)
            
            if print_output == True:
                print('RandomForestRegressor model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        f=(path+'Model_RandomForestRegressor.sav')
        pickle.dump(RFR, open(f, 'wb'))   
        np.savetxt(path+'Accumelated_points_Model_RandomForestRegressor.txt',X) 
        np.savetxt(path+'HiggsSignals_chiSquar_Model_RandomForestRegressor.txt',obs1)    
        Yellow('''\t Model saved in ./results/Model_RandomForestRegressor.sav
        Accumelated points saved in ./results/Accumelated_points_Model_RandomForestRegressor.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_RandomForestRegressor.txt ''')    
        return 


    def run_SVR_RBF(self, C=100, gamma=0.1,epsilon=0.1,print_output=True):
        SVR_RBF = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        Xf,obsf=run_train(self.L1)
        SVR_RBF.fit(Xf, obsf)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(self.iteration):
            x = generate_init_HEP(self.L)
            pred = SVR_RBF.predict(x).flatten()
            xsel = x[pred < self.th_value]           
            xsel1 = np.append(xsel[:round(self.K*(1-self.frac))],x[-round(self.K*self.frac):],axis=0)
            xsel2,obs2 = run_loop(xsel1)
            X = np.append(X, xsel2, axis=0)
            obs1 = np.append(obs1, obs2)
            if len(X) == 0: sys.exit('No accumlated points found for the first run, please try different scan ranges.')
            if (q%self.period==0 or q+1 == self.iteration):
                SVR_RBF.fit(X, obs1)
            else:
                SVR_RBF.fit(xsel2, obs2)        
            if print_output == True:
                print('SVR with rbf kernel model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        f=(path+'Model_SVR_rbf.sav')
        pickle.dump(SVR_RBF, open(f, 'wb'))        
        np.savetxt(path+'Accumelated_points_Model_SVR_rbf.txt',X) 
        np.savetxt(path+'HiggsSignals_chiSquar_Model_SVR_rbf.txt',obs1)    
        Yellow('''\t Model saved in ./results/Model_SVR_rbf.sav
        Accumelated points saved in ./results/Accumelated_points_Model_SVR_rbf.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_SVR_rbf.txt ''')        
                
        return 


    def run_SVR_POLY(self,degree=3, C=100, gamma=0.1, epsilon=0.1,print_output=True):
        SVR_POLY = SVR(kernel="poly", degree=degree, C=C, gamma=gamma, epsilon=epsilon)
        Xf,obsf=run_train(self.L1)
        SVR_POLY.fit(Xf, obsf)
        X = np.empty(shape=[0,self.function_dim]) 
        obs1 = np.empty(shape=[0,1])
        for q in range(self.iteration):
            x = generate_init_HEP(self.L)
            pred = SVR_POLY.predict(x).flatten()
            xsel = x[pred < self.th_value]           
            xsel1 = np.append(xsel[:round(self.K*(1-self.frac))],x[-round(self.K*self.frac):],axis=0)
            xsel2,obs2 = run_loop(xsel1)
            X = np.append(X, xsel2, axis=0)
            obs1 = np.append(obs1, obs2)
            if len(X) == 0: sys.exit('No accumlated points found for the first run, please try different scan ranges.')
            if (q%self.period==0 or q+1 == self.iteration):
                SVR_POLY.fit(X, obs1)
            else:
                SVR_POLY.fit(xsel2, obs2)
            if print_output == True:
                print('SVR with poly kernel  model- Run Number {} - Number of collected points= {}'.format(q,len(X)))
        f=(path+'Model_SVR_ploy.sav')
        pickle.dump(SVR_POLY, open(f, 'wb'))         
        np.savetxt(path+'Accumelated_points_Model_SVR_poly.txt',X) 
        np.savetxt(path+'HiggsSignals_chiSquar_Model_SVR_poly.txt',obs1)    
        Yellow('''\t Model saved in ./results/Model_SVR_poly.sav
        Accumelated points saved in ./results/Accumelated_points_Model_SVR_poly.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_SVR_poly.txt ''')        
                    
        return 
    


    def run_DNNC(self,neurons = 100,activation='relu',epoch=250,batch_size=500,print_output=True):
        dnn = Sequential()
        dnn.add(Dense(neurons, input_shape=(self.function_dim,)))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(neurons, activation=activation))
        dnn.add(Dense(1,activation='sigmoid'))

        dnn.compile(optimizer='adam', loss='BinaryCrossentropy')
        #RFC =RandomForestClassifier(n_estimators=300,max_depth=100)
        A=[]
        p1=Pool(processes=N_Cores)      
        A.append(p1.map(run_train_p,range(1,N_Cores+1),chunksize=1)   )
        A=np.array(A).reshape(N_Cores,2).T   
        Xfx=np.array(flatten(A[0]))
        ob1x=np.array(flatten(A[1]))
        data = np.loadtxt('good_points.txt')
        Xf = np.append(Xfx,data[:5000,:7],axis=0)     #Xf[obs1f<self.th_value]
        ob1 = np.append(ob1x,data[:5000,-1])   #obs1f[obs1f<self.th_value]
        obs1f = self.labeler(ob1,self.th_value)
        chi = ob1[obs1f==1]
        dnn.fit(Xf, obs1f, epochs=epoch,batch_size=batch_size,verbose=1)
        #RFC.fit(Xf,obs1f)
        X_g = Xf[obs1f==1]    # It is good to have some initial guide
        obs1_g = obs1f[obs1f==1]
        X_b =Xf[obs1f==0][:len(obs1_g)] 
        obs1_b =obs1f[obs1f==0][:(len(obs1_g))] 
        if len(X_g) < self.function_dim: sys.exit('First training cannot find any good points, please try different size..')
        q=0
        #for q in range(self.iteration):
        while len(obs1_g) < 20000:
            q+=1
            x = generate_init_HEP(self.L)
            #pred = RFC.predict(x).flatten()
            pred = dnn.predict(x).flatten()
            qs= np.argsort(pred)[::-1]
            xsel1 = x[qs][:round(self.K*(1-self.frac))]
            xsel1 = np.append(xsel1,x[:round(self.K*(self.frac))],axis=0)
            xsel2,obs2 = run_loop1(xsel1)
            ob = self.labeler(obs2,self.th_value)
            effc=[]
            if len(ob[ob==1])<=self.K_smote:
                Y_smote = np.append(np.zeros(len(ob[ob==0])),np.ones(len(X_g[-int(self.function_dim):])),axis=0) 
                X_smote = np.append(x[:len(ob[ob==0])],X_g[np.random.randint(len(X_g),size=self.function_dim),:],axis=0)
                oversample = SMOTE(k_neighbors=self.K_smote,n_jobs=N_Cores)
                X_over,Y_over = oversample.fit_resample(X_smote,Y_smote)
                XS_new ,ob_smote1 = run_loop1(X_over)
                ob_smote = self.labeler(ob_smote1,self.th_value)
                eff = len(ob_smote[ob_smote==1])/len(X_over[Y_over==1])
                effc.append(eff)
                print('Effeciency =  ', eff)
                chi = np.append(chi, ob_smote1[ob_smote==1])
                X_g = np.append(X_g,XS_new[ob_smote==1],axis=0)
                obs1_g = np.append(obs1_g,ob_smote[ob_smote==1],axis=0)
                X_b = np.append(X_b,xsel2[ob==0][:len(ob_smote[ob_smote==1])],axis=0)
                obs1_b = np.append(obs1_b,ob[ob==0][:len(ob_smote[ob_smote==1])],axis=0)  
                X = np.concatenate([XS_new[ob_smote==1],xsel2[ob==0][:len(ob_smote[ob_smote==1])]],axis=0)
                obs=np.concatenate([ob_smote[ob_smote==1],ob[ob==0][:len(ob_smote[ob_smote==1])]],axis=0)
            elif (len(ob[ob==1])<0.5*round(self.K*(1-self.frac)) and len(ob[ob==1])>self.function_dim):
                Y_smote = np.append(np.zeros(len(ob[ob==0])),np.ones(len(ob[ob==1])),axis=0)    
                X_smote = np.append(x[:len(ob[ob==0])],xsel2[ob==1],axis=0)
                oversample = SMOTE(k_neighbors=self.K_smote,n_jobs=N_Cores)
                X_over,Y_over = oversample.fit_resample(X_smote,Y_smote)
                XS_new ,ob_smote1 = run_loop1(X_over)
                ob_smote = self.labeler(ob_smote1,self.th_value)
                eff = len(ob_smote[ob_smote==1])/len(X_over[Y_over==1])
                effc.append(eff)
                print('Effeciency =  ', eff)
                chi = np.append(chi, ob_smote1[ob_smote==1])
                X_g = np.append(X_g,XS_new[ob_smote==1],axis=0)
                obs1_g = np.append(obs1_g,ob_smote[ob_smote==1],axis=0)
                X_b = np.append(X_b,xsel2[ob==0][:len(ob_smote[ob_smote==1])],axis=0)
                obs1_b = np.append(obs1_b,ob[ob==0][:len(ob_smote[ob_smote==1])],axis=0)  
                X = np.concatenate([XS_new[ob_smote==1],xsel2[ob==0][:len(ob_smote[ob_smote==1])]],axis=0)
                obs=np.concatenate([ob_smote[ob_smote==1],ob[ob==0][:len(ob_smote[ob_smote==1])]],axis=0)
                
            else:    
                chi = np.append(chi, obs2[ob==1])
                X_g = np.append(X_g,xsel2[ob==1],axis=0)
                obs1_g = np.append(obs1_g,ob[ob==1],axis=0)
                X_b = np.append(X_b,xsel2[ob==0][:len(xsel2[ob==1])],axis=0)
                obs1_b = np.append(obs1_b,ob[ob==0][:len(xsel2[ob==1])],axis=0)
                X = np.concatenate([xsel2[ob==1],xsel2[ob==0]],axis=0)
                obs=np.concatenate([ob[ob==1],ob[ob==0]],axis=0)
                eff = len(ob[ob==1])/len(obs2[obs2<self.th_value])
                effc.append(eff)
                print('Effeciency =  ', eff)
            #if (q%self.period==0 or q+1 == self.iteration):
            X = np.concatenate([X_g,X_b[:len(X_g)]],axis=0)
            obs = np.concatenate([obs1_g,obs1_b[:len(obs1_g)]],axis=0)
            X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
            if X_shuffled.shape[0]==0:  continue
            dnn.fit(X_shuffled,Y_shuffled,epochs=epoch,batch_size=batch_size, verbose=0)  
            #RFC.fit(X_shuffled, Y_shuffled)

            #else:
            #    X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
            #    if X_shuffled.shape[0]==0:  continue
                #dnn.fit(X_shuffled, Y_shuffled,epochs=epoch,batch_size=batch_size, verbose=0)
            #    RFC.fit(X_shuffled, Y_shuffled)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))

        #dnn.save(path+'Mode_dnn_classifier.sav')
        np.savetxt(path+'Accumelated_points_Model_dnn_classifier.txt',X_g)
        np.savetxt(path+'Eff_Model_dnn_classifier.txt',effc)
        np.savetxt(path+'HiggsSignals_chiSquar_Model_dnn_classifier.txt',chi)    
        Yellow('''\t Model saved in ./results/Model_dnn_classifier.sav
        Accumelated points saved in ./results/Accumelated_points_Model_dnn_classifier.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_dnn_classifier.txt ''')
        return




    def run_RFC(self,learning_rate=0.01,n_estimators=300,max_depth=50,print_output=True):
        RFC =RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        Xf,ob1=run_train(self.L1)
        obs1f = self.labeler(ob1,self.th_value)
        chi = ob1[obs1f==1]
        RFC.fit(Xf, obs1f)
        X_g = Xf[obs1f==1]    # It is good to have some initial guide
        obs1_g = obs1f[obs1f==1]
        X_b =Xf[obs1f==0][:len(obs1_g)]
        obs1_b =obs1f[obs1f==0][:(len(obs1_g))]
        for q in range(self.iteration):
            x = generate_init_HEP(self.L)
            pred = RFC.predict(x).flatten()
            qs = np.argsort(pred)[::-1]
            if len(x[pred>0.9]) > round(self.K*self.frac): # How to choose the good points
                xsel1 = x[pred>0.9][:round(self.K*(1-self.frac))]
            else:
                xsel1 = x[qs][:round(self.K*(1-self.frac))]
            xsel1 = np.append(xsel1,x[:round(self.K*(self.frac))],axis=0)
            xsel2,obs2 = run_loop(xsel1)
            ob = self.labeler(obs2,self.th_value)
            chi = np.append(chi, obs2[ob==1])
            X_g = np.append(X_g,xsel2[ob==1],axis=0)
            obs1_g = np.append(obs1_g,ob[ob==1],axis=0)
            X_b = np.append(X_b,xsel2[ob==0],axis=0)
            obs1_b = np.append(obs1_b,ob[ob==0],axis=0)
            if (q%self.period==0 or q+1 == self.iteration):
                X = np.concatenate([X_g,X_b],axis=0)
                obs = np.concatenate([obs1_g,obs1_b],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                RFC.fit(X_shuffled, Y_shuffled)

            else:
                X = np.concatenate([xsel2[ob==1],xsel2[ob==0]],axis=0)
                obs=np.concatenate([ob[ob==1],ob[ob==0]],axis=0)
                X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, obs)
                RFC.fit(X_shuffled, Y_shuffled)
            if print_output == True:
                print('DNN_model- Run Number {} - Number of collected points= {}'.format(q,len(X)))

        f=(path+'Model_RandomForest_classifier.sav')
        pickle.dump(RFC, open(f, 'wb'))
        np.savetxt(path+'Accumelated_points_Model_RandomForest_classifier.txt',X_g)
        np.savetxt(path+'HiggsSignals_chiSquar_Model_RandomForest_classifier.txt',chi)
        Yellow('''\t Model saved in ./results/Model_RandomForest_classifier.sav
        Accumelated points saved in ./results/Accumelated_points_Model_RandomForest_classifier.txt
        HiggsSignals chi squared saved in ./results/HiggsSignals_chiSquar_Model_RandomForest_classifier.txt ''')
        return
###############################################################
# Create an instanace of the calss to access all functions    #
###############################################################
model = scan(N_iteration,Size_initial_batch,Size_batch_L,Size_batch_K , TotVarScanned ,Chi_square_threshold ,Full_train_period,Frac_random_points,K_SMOOTE)
########## Run specific ML model #################
if args==l[0]:model.run_DNNR()
if args==l[1]:model.run_GBR()
if args==l[2]:model.run_RFR()
if args==l[3]:model.run_SVR_RBF()
if args==l[4]:model.run_SVR_POLY()
if args==l[5]:model.run_DNNC()
if args==l[6]:model.run_RFC()

######################################
if os.path.exists(str(os.getcwd())+'/work/'):
  shutil.rmtree(str(os.getcwd())+'/work/')


