import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV ,cross_validate
import time
from sklearn import metrics 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt 
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV



def Result(end , start , y_pred  , y_test ,predictModel ):
    time_learn = end - start
    acc = (metrics.r2_score(y_test , y_pred ))*100 #la qualité de modele de regression
    mean_squared_Error = mean_squared_error( y_test , y_pred )
    erreur = abs ( y_pred - y_test ).mean()

    return time_learn , acc , mean_squared_Error , erreur , y_pred , predictModel





########## ________________SVM :                   ####################################################################################
def SVM_model (x_train , x_test , y_train  , y_test ):

    svm= SVR(epsilon=0.001 , C = 100 )
    Multi_SVM = MultiOutputRegressor(svm)

    start_svm = time.process_time()
    SVM_Result = Multi_SVM.fit(x_train , y_train)
    end_svm = time.process_time()

    y_pred_svm = SVM_Result.predict(x_test)

    return  Result(end_svm , start_svm , y_pred_svm  , y_test , SVM_Result )


########## ________________ANN      _____ lbfgs :                   #############################################################################
def ANN_lbfgs_model( x_train , x_test , y_train  , y_test ):
    lbfgs = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 100))
    start_lbfgs = time.process_time()
    lbfgs.fit(x_train , y_train)
    end_lbfgs = time.process_time()

    y_pred_lbfgs = lbfgs.predict(x_test)

    return  Result( end_lbfgs , start_lbfgs  , y_pred_lbfgs  , y_test , lbfgs )


########## ________________ANN       _____ adam :                   #######################################################################
def ANN_adam_model( x_train , x_test , y_train  , y_test ):
    adam = MLPRegressor(solver='adam', alpha = 1e-5, hidden_layer_sizes=(1000, 100))

    start_adam = time.process_time()
    adam.fit(x_train , y_train)
    end_adam = time.process_time()

    y_pred_adam = adam.predict(x_test)

    return  Result( end_adam ,start_adam  , y_pred_adam  , y_test , adam )



########## ________________RFR :                   #######################################################################
def RFR_model( x_train , x_test , y_train  , y_test ):
    RandomForestRegModel = RandomForestRegressor(n_estimators=500)

    start_RF = time.process_time()
    RandomForestRegModel.fit(x_train , y_train)
    end_RF = time.process_time()

    y_pred_RandomForestReg = RandomForestRegModel.predict(x_test)

    return  Result( end_RF , start_RF ,  y_pred_RandomForestReg  , y_test , RandomForestRegModel )


########## ________________GBR :                   #######################################################################
def GBR_model( x_train , x_test , y_train  , y_test ):
    GBM = GradientBoostingRegressor()

    Multi_GBM = MultiOutputRegressor(GBM)
    start_GBM = time.process_time()
    Multi_GBM.fit(x_train , y_train)
    end_GBM = time.process_time()

    y_pred_GBM = Multi_GBM.predict(x_test)

    return  Result( end_GBM ,start_GBM , y_pred_GBM  , y_test  , Multi_GBM)


########## ________________HGBM :                   #######################################################################
def HGBR_model( x_train , x_test , y_train  , y_test ):
    HGBR = HistGradientBoostingRegressor()

    multi_HGBR= MultiOutputRegressor(HGBR)
    start_HGBR = time.process_time()
    multi_HGBR.fit(x_train , y_train)
    end_HGBR = time.process_time()

    y_pred_HGBR = multi_HGBR.predict(x_test)

    return  Result( end_HGBR ,start_HGBR ,  y_pred_HGBR  , y_test , multi_HGBR)
    
    
########## ________________XGBR :                   #######################################################################
# def XGBR_model( x_train , x_test , y_train  , y_test ):
#     XGBR = XGBRegressor()

#     multi_XGBR= MultiOutputRegressor(XGBR)
#     start_XGBR = time.process_time()
#     multi_XGBR.fit(x_train , y_train)
#     end_XGBR = time.process_time()

#     y_pred_XGBR = multi_XGBR.predict(x_test)

#     return  Result( end_XGBR ,start_XGBR ,  y_pred_XGBR  , y_test , multi_XGBR )


########## ________________VOTING :                   #######################################################################


def VOTING_model( x_train , x_test , y_train  , y_test ):
    reg1 = RandomForestRegressor(n_estimators=1000)
    reg2 = GradientBoostingRegressor()
    reg3 = SVR(epsilon=0.001 , C = 300 )
    reg4 = HistGradientBoostingRegressor()
    # reg5 = XGBRegressor() 


    ereg = VotingRegressor(estimators=[('rf', reg1) , ( 'gb', reg2), ('svm', reg3) ,   ('HGBR', reg4)])
    voting=MultiOutputRegressor(ereg)
    
    start_voting = time.time()
    ereg = voting.fit( x_train , y_train )
    end_voting = time.time()

    y_pred_voting = voting.predict(x_test)

    return  Result( end_voting ,start_voting ,  y_pred_voting  , y_test , voting )

