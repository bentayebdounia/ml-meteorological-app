import streamlit as st
from plot import *
from  MLAlgorithms import *
from DeepLearning import *
from prediction import *

def Affichage_Result(time_learn , acc , mean_squared_error , erreur):
    st.write("Accuracy=" , acc ," %" ) 
    st.write ("Time of learning= ", time_learn , " second")
    st.write("Mean square errors = " , mean_squared_error )
    st.write("Mean errors of each parameter : \n " , erreur )

def Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos):
    st.write(" ### Plot of actual and predicted maximum temperature: ")
    st.pyplot(fig_Temp_max)
    
    st.write(" ### Plot of actual and predicted minimum temperature: ")
    st.pyplot(fig_Temp_min)
    
    st.write(" ### Plot of actual and predicted wind speed: ")
    st.pyplot(fig_Vitesse_vent)
    
    st.write(" ### Plot of actual and predicted wind temperature: ")
    st.pyplot(fig_Temp_vent)
    
    st.write(" ### Plot of actual and predicted precipitation: ")
    st.pyplot(fig_Precipitation)
    
    st.write(" ### Plot of actual and predicted humidity: ")
    st.pyplot(fig_Humidite)
    
    st.write(" ### Plot of actual and predicted visibility: ")
    st.pyplot(fig_Visibilite)
    
    st.write(" ### Plot of actual and predicted Cloud cover: ")
    st.pyplot(fig_CouverNuag)
    
    st.write(" ### Plot of actual and predicted heat index: ")
    st.pyplot(fig_indice_chaleur)
    
    st.write(" ### Plot of actual and predicted Dew point: ")
    st.pyplot(fig_PointRos)


def Afficahge_predict(modelPredict):
    x_pred=Predict()
    
    #st.write(x_pred )

    Re= modelPredict.predict(x_pred)
    
    #st.write(Re )
    st.write('Maximum temperature = ',Re[0,0] ,'°')
    st.write('Minimum temperature = ',Re[0,1] ,'°')
    st.write('Wind speed = ',Re[0,2] ,'km/h')
    st.write('Wind temperature = ',Re[0,3] ,'°')
    st.write('Precipitation = ',Re[0,4] ,'mm')
    st.write('Humidity = ',Re[0,5] ,'%')
    st.write('Visibility = ',Re[0,6] ,'km')
    st.write('Cloud cover = ',Re[0,7] ,'%')
    st.write('Heat index = ',Re[0,8] ,'')
    st.write('Dew point = ',Re[0,9] ,'°')
    

def AlgoSVM(x_train , x_test ,  y_train , y_test ,y_test1 , y1  ):
    time_learn , acc_svm , mean_squared_error_svm , erreur_svm , y_pred, modelPredict = SVM_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_svm , mean_squared_error_svm , erreur_svm  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    # plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)


def AlgoANN_lbfgs(x_train , x_test ,  y_train , y_test ,y_test1 , y1   ):
    time_learn , acc_ANN_lbfgs , mean_squared_error_ANN_lbfgs , erreur_ANN_lbfgs , y_pred , modelPredict = ANN_lbfgs_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_ANN_lbfgs , mean_squared_error_ANN_lbfgs , erreur_ANN_lbfgs  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    #plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)
   


def AlgoANN_adam(x_train , x_test ,  y_train , y_test ,y_test1 , y1  ):
    time_learn , acc_adam , mean_squared_error_adam , erreur_adam , y_pred , modelPredict = ANN_adam_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_adam , mean_squared_error_adam , erreur_adam  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    #plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)



def AlgoRFR(x_train , x_test ,  y_train , y_test ,y_test1 , y1 ):
    time_learn , acc_RFR , mean_squared_error_RFR , erreur_RFR , y_pred , modelPredict = RFR_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_RFR , mean_squared_error_RFR , erreur_RFR  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    #plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)

def AlgoGBR(x_train , x_test ,  y_train , y_test ,y_test1 , y1  ):
    time_learn , acc_GBR , mean_squared_error_GBR , erreur_GBR , y_pred , modelPredict = GBR_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_GBR , mean_squared_error_GBR , erreur_GBR  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    #plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)


def Algo_HGBR(x_train , x_test ,  y_train , y_test ,y_test1 , y1  ):
    time_learn , acc_HGBR , mean_squared_error_HGBR , erreur_HGBR , y_pred , modelPredict= HGBR_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_HGBR , mean_squared_error_HGBR , erreur_HGBR  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    #plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)


# def Algo_XGBR(x_train , x_test ,  y_train , y_test ,y_test1 , y1  ):
#     time_learn , acc_XGBR , mean_squared_error_XGBR , erreur_XGBR , y_pred ,modelPredict  = XGBR_model(x_train , x_test ,  y_train , y_test)
#     Affichage_Result( time_learn  , acc_XGBR , mean_squared_error_XGBR , erreur_XGBR  )
#     fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
#     plt.show()
#     Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
#     Afficahge_predict(modelPredict)
    


def AlgoVoting(x_train , x_test ,  y_train , y_test ,y_test1 , y1  ):
    time_learn , acc_Voting , mean_squared_error_Voting , erreur_Voting , y_pred , modelPredict = VOTING_model(x_train , x_test ,  y_train , y_test)
    Affichage_Result( time_learn  , acc_Voting , mean_squared_error_Voting , erreur_Voting  )
    fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
    plt.show()
    Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
    # Afficahge_predict(modelPredict)




def Affichage_Result2(time_learn   , erreur):
    st.write ( "Time of learning= ", time_learn , " second")
    st.write("Mean of the errors of each parameter : \n " , erreur )

# def AlgoSequentiel(x_train , x_test ,  y_train , y_test , y_test1 , y1 ):
#     time_learn , acc_seq , mean_squared_Error_seq ,  erreur_seq , y_pred , modelPredict = Sequentiel_model(x_train , x_test ,  y_train , y_test)
    
#     Affichage_Result( time_learn , acc_seq , mean_squared_Error_seq ,  erreur_seq )
#     fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos = PLOTS(y_test1 , y_pred , y1)
#     plt.show()
#     Plots_result (fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos)
#     Afficahge_predict(modelPredict)

