# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn import metrics 
from sklearn.metrics import mean_squared_error 
import time

def Result(end , start , y_pred  , y_test ,predictModel ):
    time_learn = end - start
    erreur = abs ( y_pred - y_test ).mean()
    return time_learn , erreur , y_pred , predictModel

def Result2(end , start , y_pred  , y_test ,predictModel ):
    time_learn = end - start
    acc = (metrics.r2_score(y_test , y_pred ))*100 #la qualité de modele de regression
    mean_squared_Error = mean_squared_error( y_test , y_pred )
    erreur = abs ( y_pred - y_test ).mean()
    return time_learn , acc , mean_squared_Error , erreur , y_pred , predictModel

# def get_model():
#     DL_seq = Sequential()
#     DL_seq.add(Dense(40, input_dim=40, kernel_initializer='he_uniform', activation='relu'))
#     DL_seq.add(Dense(10))
#     DL_seq.compile(loss='mae', optimizer='adam')
#     return DL_seq
  

# def Sequentiel_model(x_train , x_test ,  y_train , y_test):

#   DL_seq = get_model()
#   start_seq = time.process_time()
#   DL_seq.fit(x_train, y_train, verbose=0, epochs=300)
#   end_seq = time.process_time()
#   y_pred_DL = DL_seq.predict(x_test)

#   return Result2(end_seq ,start_seq , y_pred_DL ,y_test , DL_seq)