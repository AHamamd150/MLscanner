VarMax =[] 
VarMin =[] 
VarLabel =[]
VarNum =[]
############################
# Please do not change the above tags  #
############################
pathS ='/scratch/Hammad/scanner_THDM/SPheno-4.0.5_1' #Path to SPheno directory betweenthe quotes
Lesh ='LesHouches.in.THDMII_low'     
SPHENOMODEL ='THDMII'                   # Model Name as in spheno bin directory, like SPhenoMSSM, SPheno BLSSM, etc
#################################

VarMin.append(3.00000E-01)     #Minimum value 
VarMax.append(4.00000E-01)     #Maximum value
VarLabel.append('# Lambda1Input')   # Label of the parameter in the LesHouches file
VarNum.append('1')                       # Number of the parameter in the LesHouches file


VarMin.append(1.00000E-01)     
VarMax.append(2.00000E-01)     
VarLabel.append('# Lambda2Input')         
VarNum.append('2')


VarMin.append(-5.00000E-02)     
VarMax.append(-5.00000E-01)     
VarLabel.append('# Lambda3Input')         
VarNum.append('3')

VarMin.append(6.00000E+00)     
VarMax.append(6.500000E+00)     
VarLabel.append('# Lambda4Input')         
VarNum.append('4')

VarMin.append(-6.500000E+00)     
VarMax.append(-6.300000E+00)     
VarLabel.append('# Lambda5Input')         
VarNum.append('5')


#VarMin.append(-2.50030000E+03)     
#VarMax.append(-2.50030000E+03)     
#VarLabel.append('# M12input')         
#VarNum.append('9')



VarMin.append(3.00000E+01)     
VarMax.append(3.500000E+01)     
VarLabel.append('# TanBeta')         
VarNum.append('10')

TotVarScanned = 6  # Total number of variables scanned. 
########################## Higgs Bounds #####################################
HiggsBounds = 1     # if 1 ---> switch it on. if 0 swithch off 
HBPath = '/scratch/Hammad/scanner_THDM/HiggsBounds-4.2.1' ## Full path to Higgs bounds
NH     = '3'       # Number of neutral Higgs bosons 
NCH    = '1'       # Number of charged Higgs bosons 
ExcludeHB = 1      # if 1 --> remove the points excluded by HB. If 0 ---> keep the excluded points and continue with HS
######################## Higgs Signals #######################################
HiggsSignal = 1 # if 1 ---> switch it on. if 0 swithch off 
HSPath =  '/scratch/Hammad/scanner_THDM/HiggsSignals-1.4.0'  ## Full path to Higgs bounds
PATHMHUNCERTINITY = '/scratch/Hammad/scanner_THDM/ScannerHEP/MHall_uncertainties.dat' ## mass_uncertinity file needed by higgssignals
