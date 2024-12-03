#!/usr/bin/python
import os
import random
import sys
import shutil
from scan_input import * 
from os import remove
from shutil import move
path = os.getcwd()
#paths = str(pathS)+'/'
import numpy as np
from tqdm import tqdm    
from os import remove
import glob
import multiprocessing
from multiprocessing import Pool
path = os.getcwd()      
import time 
###############################################      
def const_HB(hb):
    f1 = open(str(hb))
    for line in f1:
       if str('  # HBresult') in line:
          ohb1= line.rsplit()
    return  float(ohb1[2])
###############################################      
def const_HS1(hs):
    
    fs2 = open(str(hs))
    for line in fs2:
        if str('  # chi^2 (total)') in line:
            ohs1= line.rsplit()
    return float(ohs1[1])

def const_HS2(hs):
    fs2 = open(str(hs))
    nhs = fs2.readlines()
    mhs = nhs[-1]       
    ohs2 = mhs.rsplit() 
    return float(ohs2[-5])    
##############Checks######################         
if not os.path.exists(str(pathS)+'/'+('bin/SPheno')):
    sys.exit ('"/bin/SPheno" NOT EXIST, PLEASE TYPE make.')
if not  os.path.exists(str(pathS)+'/'+str(Lesh)):
    sys.exit (str(pathS)+'/'+str(Lesh)+' NOT EXIST.')
if not  os.path.exists(str(pathS)+'/'+'/bin/SPheno%s'%(str(SPHENOMODEL))):
    sys.exit (str(pathS)+'/'+'/bin/SPheno%s'%(str(SPHENOMODEL))+' NOT EXIST.')
#############################################
def replace_HS(i):
    root=open (str(i))
    f = open ('fout','w+')
    for y in root :
        if str ('Block HiggsCouplingsFermions, #') in y :
            x = y.rsplit()
            y = y.replace('Block HiggsCouplingsFermions, #', 'Block HiggsBoundsInputHiggsCouplingsFermions #')
        if str ('Block HiggsCouplingsBosons #') in y: 
            x = y.rsplit()
            y = y.replace('Block HiggsCouplingsBosons #','Block HiggsBoundsInputHiggsCouplingsBosons #' )
        f.write(y)
    os.rename('fout',str(i))
    f.close()

############Generate random numbers##################
def generate_init_HEP(n):
    AI_2 = np.empty(shape=[0,TotVarScanned])
    for i in range(n):
        LHEfile = open(str(pathS)+'/'+str(Lesh),'r+')
        AI_1 = []
        for line in LHEfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = VarMin[yy] + (VarMax[yy] - VarMin[yy])*random.random()
                    AI_1.append(value)
       
        AI_1= np.array(AI_1).reshape(1,TotVarScanned)   
        AI_2 = np.append(AI_2,AI_1,axis=0)    
    return AI_2         
#######################################
def run_train(npoints,paths,seed):
    random.seed(seed)
    #os.chdir(paths)
    
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
    ###########################
    for xx in range(0,npoints):
        #print('Calculated likelihood points= %s'%str(xx))
        newrunfile = open('newfile','w')
        oldrunfile = open(str(Lesh),'r+')
        AI_L = []
        for line in oldrunfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = VarMin[yy] + (VarMax[yy] - VarMin[yy])*random.random()
                    AI_L.append(value)
                    valuestr = str("%.4E" % value)
                    newrunfile.write(VarNum[yy]+'   '+valuestr +str('     ')+ VarLabel[yy]+'\n')
                    NewlineAdded = 1
            if NewlineAdded == 0:
                newrunfile.write(line)
        newrunfile.close()
        oldrunfile.close()
        os.remove(str(Lesh))
        AI_L= np.array(AI_L).reshape(1,TotVarScanned)
        ############################    
        os.rename('newfile',str(Lesh))
        os.system('./bin/SPheno'+str(SPHENOMODEL)+' '+str(Lesh)+' spc.slha'+' >  out.txt')
        out = open(str(paths)+'out.txt','r+')
        for l in out:
            #print l
            if str('Finished!') in l:
                    
                if (HiggsBounds != 0 ):
                    print('Run HiggsBounds= %s'%str(xx))
                    if os.path.exists(str(paths)+'spc.slha'):
                        os.system('eval  %s/HiggsBounds LandH SLHA  %s %s %s/spc.slha >/dev/null'%(str(HBPath),str(NH),str(NCH),str(paths))) 
                        AI_hb=const_HB(str(paths)+'spc.slha')  
                        if AI_hb == 0 : 
                           continue                       

                if (HiggsSignal ==1):
                    print('Run HiggsSignals')
                    if os.path.exists(str(paths)+'spc.slha'):
                        replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals %s peak 2 SLHA  %s %s %s/spc.slha >/dev/null'%(str(HSPath),str(LHC_run),str(NH),str(NCH),str(paths)))
                        #os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(pathS),str(pathS),str(pathS)))
                        #os.rename(str(pathS)+'/sps2.out',str(pathS)+'/spc.slha')
                        AI_yv=const_HS1(str(paths)+'spc.slha')   
                        AI_Y.append(AI_yv)  
                       
                if (HiggsSignal ==2):
                    print('Run HiggsSignals')
                    if os.path.exists(str(paths)+'spc.slha'):
                        replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals %s peak 2 effC  %s %s %s/ >/dev/null'%(str(HSPath),str(LHC_run),str(NH),str(NCH),str(paths)))
                        os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(paths),str(paths),str(paths)))
                        os.rename(str(paths)+'/sps2.out',str(paths)+'/spc.slha')
                        AI_yv=const_HS2(str(paths)+'spc.slha')
                        AI_Y.append(AI_yv)
                        #if AI_yv < Chi_square_threshold:
                            
                           #os.system('cp -rf %s/spc.slha ./spec/spc_%s.slha'%(str(time.time()),str(paths)))          
                AI_X = np.append(AI_X,AI_L,axis=0)
        os.remove('out.txt')
    os.chdir(path)
    return np.array(AI_X),np.array(AI_Y)

############################################################################################ 
############################################################################################
###########################################################################################
def run_loop(npoints,paths,seed):
    #paths = str(pathS)+'/'
    #os.chdir(paths)
    random.seed(seed)
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
    ###########################
    for xx in range(0,npoints.shape[0]):
        newrunfile = open('newfile','w')
        oldrunfile = open(str(Lesh),'r+')
        AI_L = []
        for line in oldrunfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = npoints[xx,yy]
                    AI_L.append(value)
                    valuestr = str("%.4E" % value)
                    newrunfile.write(VarNum[yy]+'   '+valuestr +str('     ')+ VarLabel[yy]+'\n')
                    NewlineAdded = 1
            if NewlineAdded == 0:
                newrunfile.write(line)
        newrunfile.close()
        oldrunfile.close()
        os.remove(str(Lesh))
        AI_L= np.array(AI_L).reshape(1,TotVarScanned)
        ############################    
        os.rename('newfile',str(Lesh))
        os.system('./bin/SPheno'+str(SPHENOMODEL)+' '+str(Lesh)+' spc.slha'+' >  out.txt')
        out = open(str(paths)+'out.txt','r+')
        for l in out:
            #print l
            if str('Finished!') in l:
                    
                if (HiggsBounds != 0 ):
                    print('Run HiggsBounds')
                    if os.path.exists(str(paths)+'spc.slha'):
                        os.system('eval  %s/HiggsBounds LandH SLHA  %s %s %s/spc.slha >/dev/null'%(str(HBPath),str(NH),str(NCH),str(paths))) 
                        AI_hb=const_HB(str(paths)+'spc.slha')     
                        if AI_hb == 0 : 
                           continue                       

                if (HiggsSignal ==1):
                    print('Run HiggsSignals')
                    if os.path.exists(str(paths)+'spc.slha'):
                        replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals %s peak 2 SLHA  %s %s %s/spc.slha >/dev/null'%(str(HSPath),str(LHC_run),str(NH),str(NCH),str(paths)))
                        #os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(pathS),str(pathS),str(pathS)))
                        #os.rename(str(pathS)+'/sps2.out',str(pathS)+'/spc.slha')
                        AI_yv=const_HS1(str(paths)+'spc.slha')
                        AI_Y.append(AI_yv)

                if (HiggsSignal ==2):
                    print('Run HiggsSignals')
                    if os.path.exists(str(paths)+'spc.slha'):
                        replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals %s peak 2 effC  %s %s %s/ >/dev/null'%(str(HSPath),str(LHC_run),str(NH),str(NCH),str(paths)))
                        os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(paths),str(paths),str(paths)))
                        os.rename(str(pathS)+'/sps2.out',str(paths)+'/spc.slha')
                        AI_yv=const_HS2(str(paths)+'spc.slha')
                        AI_Y.append(AI_yv)       
                          
                AI_X = np.append(AI_X,AI_L,axis=0)
        os.remove('out.txt')
    os.chdir(path)
    return np.array(AI_X),np.array(AI_Y)


###############################################
############################################################################################ 
############################################################################################
###########################################################################################
def run_loop1(npoints):
    paths = str(pathS)+'/'
    os.chdir(paths)
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
    ###########################
    for xx in tqdm(range(0,npoints.shape[0]),ascii=True,colour='green',desc='Spheno+HB+HS progress'):
        newrunfile = open('newfile','w')
        oldrunfile = open(str(Lesh),'r+')
        AI_L = []
        for line in oldrunfile: 
            NewlineAdded = 0
            for yy in range(0,TotVarScanned):
                if str(VarLabel[yy]) in line:
                    value = npoints[xx,yy]
                    AI_L.append(value)
                    valuestr = str("%.4E" % value)
                    newrunfile.write(VarNum[yy]+'   '+valuestr +str('     ')+ VarLabel[yy]+'\n')
                    NewlineAdded = 1
            if NewlineAdded == 0:
                newrunfile.write(line)
        newrunfile.close()
        oldrunfile.close()
        os.remove(str(Lesh))
        AI_L= np.array(AI_L).reshape(1,TotVarScanned)
        ############################    
        os.rename('newfile',str(Lesh))
        os.system('./bin/SPheno'+str(SPHENOMODEL)+' '+str(Lesh)+' spc.slha'+' >  out.txt')
        out = open(str(paths)+'out.txt','r+')
        for l in out:
            #print l
            if str('Finished!') in l:
                    
                if (HiggsBounds != 0 ):
                    if os.path.exists(str(paths)+'spc.slha'):
                        os.system('eval  %s/HiggsBounds LandH SLHA  %s %s %s/spc.slha >/dev/null'%(str(HBPath),str(NH),str(NCH),str(pathS))) 
                        AI_hb=const_HB(str(paths)+'spc.slha')     
                        if AI_hb == 0 : 
                           continue                       

                if (HiggsSignal ==1):
                    if os.path.exists(str(paths)+'spc.slha'):
                        replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals %s peak 2 SLHA  %s %s %s/spc.slha >/dev/null'%(str(HSPath),str(LHC_run),str(NH),str(NCH),str(pathS)))
                        #os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(pathS),str(pathS),str(pathS)))
                        #os.rename(str(pathS)+'/sps2.out',str(pathS)+'/spc.slha')
                        AI_yv=const_HS1(str(paths)+'spc.slha')
                        AI_Y.append(AI_yv)

                if (HiggsSignal ==2):
                    if os.path.exists(str(paths)+'spc.slha'):
                        replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals %s peak 2 effC  %s %s %s/ >/dev/null'%(str(HSPath),str(LHC_run),str(NH),str(NCH),str(pathS)))
                        os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(pathS),str(pathS),str(pathS)))
                        os.rename(str(pathS)+'/sps2.out',str(pathS)+'/spc.slha')
                        AI_yv=const_HS2(str(paths)+'spc.slha')
                        AI_Y.append(AI_yv)       
                        #os.system('cp -rf %s/spc.slha ./spec/spc_%s.slha'%(str(time.time()),str(paths)))
                AI_X = np.append(AI_X,AI_L,axis=0)
        os.remove('out.txt')
    os.chdir(path)
    return np.array(AI_X),np.array(AI_Y)


###############################################

    
###########################################
## Trying to prallize the initial scan ########
###############################################
def dir_prepare(i):
  
  if  not os.path.exists(str(os.getcwd())+'/work/work_%s/'%str(i)):
    os.mkdir(str(os.getcwd())+'/work/work_%s/'%str(i))
  shutil.copytree(str(pathS)+"/bin/",str(os.getcwd())+'/work/work_%s/bin/'%str(i))
  shutil.copy(str(pathS)+'/'+str(Lesh),str(os.getcwd())+'/work/work_%s/'%str(i))  
######################
def run_train_p(i):
  if os.path.exists(str(os.getcwd())+'/work/work_%s/'%str(i)):
      paths = str(os.getcwd())+'/work/work_%s/'%str(i)
      os.chdir(str(os.getcwd())+'/work/work_%s/'%str(i)) 
      A,B=run_train(round(Size_initial_batch/N_Cores),paths,i*10) 
      return np.array(A),np.array(B)     
###############################      
def flatten(t):
        return [item for sublist in t for item in sublist]
        
##############################        






























































    
