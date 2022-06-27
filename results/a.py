import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "cursive"
#####
X= np.loadtxt('Accumelated_points_Model_RandomForest_classifier.txt')
Y = np.loadtxt('HiggsSignals_chiSquar_Model_RandomForest_classifier.txt')

#X = np.loadtxt('Accumelated_points_Model_RandomForestRegressor_THDM.txt')
#Y = np.loadtxt('HiggsSignals_chiSquar_Model_RandomForestRegressor_THDM.txt')

#X  = np.loadtxt("Accumelated_points_Model_RandomForest_classifier.txt")
#Y = np.loadtxt("HiggsSignals_chiSquar_Model_RandomForest_classifier.txt")

#X = np.loadtxt("Accumelated_points_Model_dnn_Regression.txt")
#Y = np.loadtxt("HiggsSignals_chiSquar_Model_dnn_Regression.txt")

###################3

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')




p = ax.scatter(X[:,3],X[:,2],X[:,4],c = Y,s=0.5,cmap='jet');
         
#ax.scatter(x1[pred>0.9],x2[pred>0.9],x3[pred>0.9],color='b',s=1,alpha=0.3);
cbar= fig.colorbar(p,shrink=0.4)
ax.view_init(30, 50);

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.set_xlim(5,6.4)
ax.set_ylim(-0.5,-0.1)
ax.set_zlim(-6.5,-6)

#ax.set_xlabel(r'$\lambda_4$',fontsize = 15);
#ax.set_ylabel(r'$\lambda_5$',fontsize = 15);
#ax.set_zlabel(r'$\tan\beta$',fontsize = 15);
#plt.title('DNN Regressor',fontsize = 15)
plt.show()



