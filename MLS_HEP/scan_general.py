#!/usr/bin/python
import os
import random
import sys
import shutil
import cmath 
import time
from scan_input import * 
from os import remove
from shutil import move
import glob
path = os.getcwd()
paths = str(pathS)+'/'
import numpy as np
from tqdm import tqdm
        
###############################################      
def const_HB(hb):
    f1 = open(str(hb))
    for line in f1:
       if str('  # HBresult') in line:
          ohb1= line.rsplit()
    return  float(ohb1[2])
###############################################      
def const_HS(hs):
    fs2 = open(str(hs))
    nhs = fs2.readlines()
    mhs = nhs[-1]       
    ohs2 = mhs.rsplit() 
    return float(ohs2[-5])    
##############Checks######################         
if not os.path.exists(str(paths)+('bin/SPheno')):
    sys.exit ('"/bin/SPheno" NOT EXIST, PLEASE TYPE make.')
if not  os.path.exists(str(paths)+str(Lesh)):
    sys.exit (str(paths)+str(Lesh)+' NOT EXIST.')
if not  os.path.exists(str(paths)+'/bin/SPheno%s'%(str(SPHENOMODEL))):
    sys.exit (str(paths)+'/bin/SPheno%s'%(str(SPHENOMODEL))+' NOT EXIST.')
############Generate random numbers##################
def generate_init_HEP(n):
    AI_2 = np.empty(shape=[0,TotVarScanned])
    for i in range(n):
        LHEfile = open(str(paths)+str(Lesh),'r+')
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
def run_train(npoints):
    os.chdir(paths)
    AI_X = np.empty(shape=[0,TotVarScanned])
    AI_Y = []
    ###########################
    for xx in tqdm(range(0,npoints),ascii=True,colour='green',desc='Spheno+HB+HS progress'):
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
                    if os.path.exists(str(paths)+'spc.slha'):
                        os.system('eval  %s/HiggsBounds LandH SLHA  %s %s %s/spc.slha >/dev/null'%(str(HBPath),str(NH),str(NCH),str(pathS))) 
                        AI_hb=const_HB(str(paths)+'spc.slha')     
                        if AI_hb == 0 : 
                           continue                       

                if (HiggsSignal != 0):
                    if os.path.exists(str(paths)+'spc.slha'):
                        #replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals latestresults peak 2 effC  %s %s %s >/dev/null'%(str(HSPath),str(NH),str(NCH),str(paths)))
                        os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(pathS),str(pathS),str(pathS)))
                        os.rename(str(pathS)+'/sps2.out',str(pathS)+'/spc.slha')
                        AI_yv=const_HS(str(paths)+'spc.slha')   
                        AI_Y.append(AI_yv)  
                       
                          
                AI_X = np.append(AI_X,AI_L,axis=0)
        os.remove('out.txt')
    os.chdir(path)
    return np.array(AI_X),np.array(AI_Y)
############################################################################################ 
############################################################################################
###########################################################################################
def run_loop(npoints):
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

                if (HiggsSignal != 0):
                    if os.path.exists(str(paths)+'spc.slha'):
                        #replace_HS(str(paths)+'spc.slha')
                        os.system('eval  %s/HiggsSignals latestresults peak 2 effC  %s %s %s >/dev/null'%(str(HSPath),str(NH),str(NCH),str(paths)))
                        os.system('cat %s/spc.slha %s/HiggsSignals_results.dat > %s/sps2.out'%(str(pathS),str(pathS),str(pathS)))
                        os.rename(str(pathS)+'/sps2.out',str(pathS)+'/spc.slha')
                        AI_yv=const_HS(str(paths)+'spc.slha')   
                        AI_Y.append(AI_yv)  
                       
                          
                AI_X = np.append(AI_X,AI_L,axis=0)
        os.remove('out.txt')
    os.chdir(path)
    return np.array(AI_X),np.array(AI_Y)



