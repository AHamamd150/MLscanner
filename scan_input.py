VarMax =[] 
VarMin =[] 
VarLabel =[]
VarNum =[]
######################################
# Please do not change the above tags#
######################################
N_iteration = 50
Size_initial_batch=100000
Size_batch_L = 100000
Size_batch_K = 200
Chi_square_threshold = 95
Full_train_period = 2
Frac_random_points = 0.1
############################################
pathS ='/home/SPheno-4.0.5_2' #Path to SPheno directory betweenthe quotes
Lesh ='LesHouches.in.THDMII_low'     
SPHENOMODEL ='THDMII'                   # Model Name as in spheno bin directory, like SPhenoMSSM, SPheno BLSSM, etc
#################################


VarMin.append(1.00000E-01)          #Minimum value 
VarMax.append(5.00000E+00)          #Maximum value
VarLabel.append('# Lambda1Input')   # Label of the parameter in the LesHouches file
VarNum.append('1')                  # Number of the parameter in the LesHouches file


VarMin.append(1.00000E-01)     
VarMax.append(2.00000E-01)     
VarLabel.append('# Lambda2Input')         
VarNum.append('2')


VarMin.append(-1.00000E+00)     
VarMax.append(-5.00000E-02)     
VarLabel.append('# Lambda3Input')         
VarNum.append('3')

VarMin.append(1.00000E+00)     
VarMax.append(10.00000E+00)     
VarLabel.append('# Lambda4Input')         
VarNum.append('4')

VarMin.append(-10.00000E+00)     
VarMax.append(0.00000E+00)     
VarLabel.append('# Lambda5Input')         
VarNum.append('5')


VarMin.append(-2.0030000E+03)     
VarMax.append(-3.0030000E+03)     
VarLabel.append('# M12input')         
VarNum.append('9')



VarMin.append(2.00000E+01)     
VarMax.append(4.000000E+01)     
VarLabel.append('# TanBeta')         
VarNum.append('10')

TotVarScanned = 7  # Total number of variables scanned. 
########################## Higgs Bounds #####################################
HiggsBounds = 1 #### if 1 ---> switch it on. if 0 swithch off 
HBPath = '/home/HiggsSignals-2.2.3beta' ## Full path to Higgs bounds
NH     = '3'       # Number of neutral Higgs bosons 
NCH    = '1'       # Number of charged Higgs bosons 
ExcludeHB = 1      # if 1 --> remove the points excluded by HB. If 0 ---> keep the excluded points 
######################## Higgs Signals #######################################
HiggsSignal = 2 #### if 1 ---> switch it on. if 0 swithch off 
HSPath =  '/home/HiggsBounds-5.3.2beta'  ## Full path to Higgs bounds
PATHMHUNCERTINITY = '/home/MHall_uncertainties.dat' ## mass_uncertinity file needed by higgssignals
LHC_run = 'LHC13'     # put the upper value of chi square total ..  if 0 do not remove any spec

