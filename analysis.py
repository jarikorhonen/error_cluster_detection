'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  Example of using analytical model and machine learning to predict 
  the visibility of packet error clusters.
  
  This example has some hardcoded values and only works with the
  data used for the original study (IEEE Transactions paper); however,
  you can use this code as an example to start the analysis with your
  own data.
  
  Written by Jari Korhonen, Shenzhen University
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Load libraries
import pandas
import random as rnd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy import optimize
from sklearn.preprocessing import MinMaxScaler
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn import ensemble
import numpy as np

# ========================================================================       
# Load data
#
array = []
filepath = 'f:/videos/features.csv'
df = pandas.read_csv(filepath, skiprows=[], header=None)
array = df.values

# Features of error clusters are the input
X = array[:,1:16]

# Subjective error visibility is the output
Y = array[:,0]

# ========================================================================
# Try the analytical model first
#
E_cl = [np.log10(X[j,1]*pow(X[j,9],2)*X[j,14]*X[j,3]) for j in range(0,len(Y))]

def piecewise_linear(x, x0, x1, e):
    return np.piecewise(x, [x<x0, (x>=x0) & (x<x1), x>=x1], [lambda x:0, lambda x:pow((x-x0)/(x1-x0),e), lambda x:1])

p1, err = optimize.curve_fit(piecewise_linear, E_cl, Y, p0=[-2,8,1.1])
Y_pred = piecewise_linear(E_cl, *p1)
E_cl_curve = [i/40 for i in range(-240,400)]
Y_curve = piecewise_linear(E_cl_curve, *p1)

print('===== Results with the analytical model ===========================')
print('PCC:  ',scipy.stats.pearsonr(Y_pred,Y)[0])
print('SRCC: ',scipy.stats.spearmanr(Y_pred,Y)[0])
print('MSE:  ',np.mean(pow(Y_pred-Y,2)))
print(' ')

# Plot the graph, if you wish
'''
plt.xlabel('Error Cluster Visibility Index E_CL')
plt.ylabel('Subjective Error Visibility')
plt.title('Subjective Error Visibility versus E_CL')
plt.xlim(-4,7)
plt.plot(E_cl,Y,'r.',E_cl_curve,Y_curve,'b-')
'''

# ========================================================================
# Try the modified analytical model with averages
#

E_cl_avg = []

Y_avg = [i/20 for i in range(0,20)]
for i in range(0,20):
    values = [E_cl[j] for j in range(0,len(Y)) if Y[j] > i/20-0.001 and Y[j] < i/20+0.001]
    E_cl_avg = E_cl_avg + [np.mean(values)]

p2, err = optimize.curve_fit(piecewise_linear, E_cl_avg, Y_avg, p0=[-2,8,1.1])
Y_avg_pred = piecewise_linear(E_cl_avg, *p1)
Y_avg_curve = piecewise_linear(E_cl_curve, *p2)

print('===== Results with the analytical model on averages ===============')
print('PCC:  ',scipy.stats.pearsonr(Y_avg_pred,Y_avg)[0])
print('SRCC: ',scipy.stats.spearmanr(Y_avg_pred,Y_avg)[0])
print('MSE:  ',np.mean(pow(Y_avg_pred-Y_avg,2)))
print(' ')

# Plot the graph, if you wish
'''
plt.xlabel('Error Cluster Visibility Index E_CL')
plt.ylabel('Subjective Error Visibility')
plt.title('Subjective Error Visibility versus E_CL')
plt.xlim(-4,7)
plt.plot(E_cl,Y,color='0.75',linestyle='none',marker='.') 
plt.plot(E_cl_avg,Y_avg,color='0',linestyle='none',marker='o') 
plt.plot(E_cl_curve,piecewise_linear(E_cl_curve, *p2),'b-')
'''


# ========================================================================
# Finally, try the machine learning approach
#

def train_and_validate(X_inp, Y_inp, seed):
    
    # To compensate inbalance, we need to define different test size on each 
    # error visibility level
    #
    testsizes=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    
    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []

    
    for i in range(0,20):
        X_this = [X_inp[j,:] for j in range(0,len(Y)) if Y[j] > i/20-0.001 and Y[j] < i/20+0.001]
        Y_this = [Y_inp[j] for j in range(0,len(Y)) if Y[j] > i/20-0.001 and Y[j] < i/20+0.001]
        X_train_this, X_validation_this, Y_train_this, Y_validation_this = model_selection.train_test_split(X_this, Y_this, test_size=testsizes[i], random_state=seed)
        
        Y_train.extend(Y_train_this)
        X_train.extend(X_train_this)
        Y_validation.extend(Y_validation_this)
        X_validation.extend(X_validation_this)        
        
    # =====================================================================
    # Regression training here. You can just use whatever regression model
    # you like.
    #
    # 
    
    # First, scale the features
    scaler = MinMaxScaler(feature_range=(0.0001, 1))
    X_train = scaler.fit_transform(X_train)   
    
    # Then, define the model and the parameters
    #model = GradientBoostingRegressor()
    #model = MLPRegressor()
    #model = SVR()
    #model = RandomForestRegressor()
    model = BaggingRegressor(base_estimator=ensemble.GradientBoostingRegressor())
    model.fit(X_train, Y_train)

    # =====================================================================
    # Validation part starts here. Nothing very special.
    #
    X_validation = scaler.transform(X_validation)
    Y_pred = model.predict(X_validation)  
    return Y_pred, Y_validation
    
    #======================================================================
    
# =========================================================================
# Try the training and validation with ten different random splits
#
pccs = []
sroccs = []
mses = []
for i in range(1,10):
    Y_pred, Y_valid = train_and_validate(X,Y,10*i)
    pccs.append(scipy.stats.pearsonr(Y_pred,Y_valid)[0])
    sroccs.append(scipy.stats.spearmanr(Y_pred,Y_valid)[0])
    mses.append(np.mean([pow(Y_pred[j]-Y_valid[j],2) for j in range(0,len(Y_pred))]))
    
print('===== Results with the machine learning model (10-fold) ===========')
print('Average PCC:  ',np.mean(pccs),' std: ', np.std(pccs))
print('Average SRCC: ',np.mean(sroccs),' std: ', np.std(sroccs))
print('Average MSE:  ',np.mean(mses),' std: ', np.std(mses))
print(' ')


# If you like, you can plot an example figure here
'''
plt.ylabel('Predicted Subjective Error Visibility')
plt.xlabel('Subjective Error Visibility')
plt.title('Predicted vs. Actual Subjective Error Visibility')

Y_pred, Y_valid = train_and_validate(X,Y,900)
plt.xlim(-0.1,1)
plt.ylim(-0.1,1)
plt.plot(Y_valid,Y_pred,'ro')
'''
# ======= End of file =====================================================



