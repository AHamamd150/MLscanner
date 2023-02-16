VarMax =[] 
VarMin =[] 
VarLabel =[]
VarNum =[]
######################################
# Please do not change the above tags#
######################################
N_Cores = 65              # Number of cores for the parallel processing in the initial random scan, "0th iteration"
N_iteration = 5           # Number of iterations 
Size_initial_batch=5000
Size_batch_L = 100000
Size_batch_K = 300
Chi_square_threshold = 95 # Chi squared threshold
Full_train_period = 1     # Train period, 1 -> each iteration, 2 each second iteration, etc
Frac_random_points = 0.2  # Fraction of the random point
K_SMOOTE = 5    # Number of k neighbours to be used by SMOOTE
############################################
pathS ='./SPheno-4.0.5_2' #Path to SPheno directory betweenthe quotes
Lesh ='LesHouches.in.THDMII_low'     
SPHENOMODEL ='THDMII'                   # Model Name as in spheno bin directory, like SPhenoMSSM, SPheno BLSSM, etc
#################################


VarMin.append(-10.00000E+01)     #Minimum value 
VarMax.append(1.00000E+01)     #Maximum value
VarLabel.append('# Lambda1Input')   # Label of the parameter in the LesHouches file
VarNum.append('1')                       # Number of the parameter in the LesHouches file


VarMin.append(-1.00000E+01)     
VarMax.append(1.00000E+01)     
VarLabel.append('# Lambda2Input')         
VarNum.append('2')


VarMin.append(-1.00000E+01)     
VarMax.append(1.00000E+01)     
VarLabel.append('# Lambda3Input')         
VarNum.append('3')

VarMin.append(-1.00000E+01)     
VarMax.append(1.00000E+01)     
VarLabel.append('# Lambda4Input')         
VarNum.append('4')

VarMin.append(-1.00000E+01)     
VarMax.append(1.00000E+01)     
VarLabel.append('# Lambda5Input')         
VarNum.append('5')


VarMin.append(-2.00000E+03)     
VarMax.append(-3.000000E+03)     
VarLabel.append('# M12input')         
VarNum.append('9')



VarMin.append(1.00000E+01)     
VarMax.append(4.000000E+01)     
VarLabel.append('# TanBeta')         
VarNum.append('10')

TotVarScanned = 7  # Total number of variables scanned. 
########################## Higgs Bounds #####################################
HiggsBounds = 1 #### if 1 ---> switch it on. if 0 swithch off 
HBPath = './HiggsBounds-4.2.1' ## Full path to Higgs bounds
NH     = '3'       # Number of neutral Higgs bosons 
NCH    = '1'       # Number of charged Higgs bosons 
ExcludeHB = 1      # if 1 --> remove the points excluded by HB. If 0 ---> keep the excluded points 
######################## Higgs Signals #######################################
HiggsSignal = 2 #### if 1 ---> switch it on. if 0 swithch off 
HSPath =  './HiggsSignals-1.4.0'  ## Full path to Higgs bounds
PATHMHUNCERTINITY = './MHall_uncertainties.dat' ## mass_uncertinity file needed by higgssignals
LHC_run = 'latestresults'     # put the upper value of chi square total ..  if 0 do not remove any spec

