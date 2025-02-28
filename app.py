import streamlit as st
from PIL import Image
from preparationData import *
from functions import *

meteo = Image.open('images/weather-icon1-67.png') 
st.set_page_config(page_title = 'Machine Learning Apply On Weather Data', page_icon=meteo )
st.title ("Machine Learning Apply On Weather Data")

data_final , X , Y = PreparationData()
x_train , x_test ,  y_train , y_test , y_test1 , y1 = TrainingAndTestDATA ( X , Y )

st.write("Weather forecasting has been one of the most interesting fields \
    and a lot of work has been done in this constraint over the years, many different \
    techniques have also been applied to predict temperature and humidity and other weather parameters, \
    this type of prediction is one of the important parts of it that can be useful to society as well as to \
    the economy. ")
st.image(Image.open('images/4_saison.jpg'))

st.write ("Data mining, machine learning, and deep learning are methods that \
    have been developed recently and can be applied successfully in this field as well.\
    For this, we applied Machine Learning techniques on a database of ", X.shape[0] ," historical data of"
    , y_train.shape[1] ,"meteorological parameters from 4 cities in western Algeria \
    (Tlemcen, Oran, Sidi bel Abbes, Mostaganem). From which we took ",y_train.shape[0], " data to train them \
    and ", y_test.shape [0] ," data to test them.")

st.write("### Describe Data Frame ")
st.write(data_final.info())
st.write(data_final.describe())

st.title("""
    Explore different algorithms
    """)
Choise = st.sidebar.radio("Select your Learning Model: " ,("Machine Learning" , "DEEP LEARNING")  )
if (Choise == "Machine Learning") :
    Name_algorithm = st.sidebar.selectbox("Select Algorithms of machine learning:", ( " " , "SVM" , "Artificial Neural Network Algorithm" , "Random Forest Regressor" , "Gradient Boosting Regressor" , "Histogram Gradient Boosting Regressor" , "X Gradient Boosting Regressor" ,  "Voting" ))

    if Name_algorithm == 'SVM' :
        st.header("""
            Support Vector Machine Algorithm 
            """)
        st.write("""Support vector regression (SVR) is a statistical method that examines the linear relationship between two continuous variables.
        In regression problems, we generally try to find a line that best fits the data provided. The equation of the line in its simplest form is described as below y=mx +c
        In the case of regression using a support vector machine, we do something similar but with a slight change. Here we define a small error value e (error = prediction - actual).

        """)
        # AlgoSVM(x_train , x_test ,  y_train , y_test , y_test1 , y1)
            

    elif Name_algorithm == 'Artificial Neural Network Algorithm' :
        st.header("""
            Artificial Neural Network Algorithm 
            """)
        st.write("""Neural networks are created by adding the layers of these perceptrons together, known as a multi-layer perceptron model. 
        There are three layers of a neural network - the input, hidden, and output layers. 
        The input layer directly receives the data, whereas the output layer creates the required output. 
        The layers in between are known as hidden layers where the intermediate computation takes place.""")
        optimaz = st.sidebar.radio("Select optimization : " ,("lbfgs" , "adam") ) 
        if(optimaz =="lbfgs") :
            st.header("""
            lbfgs optimization:
            """)
            AlgoANN_lbfgs(x_train , x_test ,  y_train , y_test , y_test1 , y1)
        else :
            st.header(""" 
            adam optimization:
            """)
            AlgoANN_adam(x_train , x_test ,  y_train , y_test , y_test1 , y1)

    elif Name_algorithm == 'Random Forest Regressor' :
        st.header("""
            Random Forest Regressor Algorithm
            """)
        st.write("""
        Random forest is a bagging technique and not a boosting technique. The trees in random forests are run in parallel. 
        There is no interaction between these trees while building the trees.
It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode 
of the classes (classification) or mean prediction (regression) of the individual trees.
        """)
        AlgoRFR(x_train , x_test ,  y_train , y_test , y_test1 , y1)

    elif Name_algorithm == 'Gradient Boosting Regressor' :
        st.header("""
            Gradient Boosting Regressor Algorithm
            """)
        st.write("""
        Gradient boosting algorithm is one of the most powerful algorithms in the field of machine learning.
         As we know that the errors in machine learning algorithms are broadly classified into two categories i.e. Bias Error and Variance Error. 
         As gradient boosting is one of the boosting algorithms it is used to minimize bias error of the model.
        """)
            
        AlgoGBR(x_train , x_test ,  y_train , y_test , y_test1 , y1)

    elif Name_algorithm == 'Histogram Gradient Boosting Regressor' :
        st.header("""
            Histogram Gradient Boosting Regressor Algorithm
            """)
        st.write(""" Training the trees that are added to the ensemble can be dramatically accelerated by discretizing (binning) 
        the continuous input variables to a few hundred unique values. Gradient boosting ensembles 
        that implement this technique and tailor the training algorithm around input variables under this
        transform are referred to as histogram-based gradient boosting ensembles.
        """) 
        Algo_HGBR(x_train , x_test ,  y_train , y_test , y_test1 , y1 )

    elif Name_algorithm == 'X Gradient Boosting Regressor' :
        st.header("""
            X Gradient Boosting Regressor Algorithm
            """)
        st.write(""" XGBoost stands for Extreme Gradient Boosting; it is a specific implementation of 
        the Gradient Boosting method which uses more accurate approximations to find the best tree model.
        It employs a number of nifty tricks that make it exceptionally successful, particularly with structured data.
        """)
            
        # Algo_XGBR(x_train , x_test ,  y_train , y_test  , y_test1 , y1 )
            

    elif Name_algorithm == 'Voting' :
        st.header("""
            VOTING Algorithm
            """)
        st.write("""A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models.
        It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble.
        A voting ensemble works by combining the predictions from multiple models. It can be used for classification or regression. In the case of regression, this involves calculating the average of the predictions from the models.""")
        AlgoVoting(x_train , x_test ,  y_train , y_test , y_test1 , y1 )    

else : 
    Name_algorithm = st.sidebar.selectbox("Select Algorithms of Deep Learning:" ,(" " , "Sequential Algorithm" ))
    if Name_algorithm == 'Sequential Algorithm' :
        st.header("""
            Sequential Algorithm 
            """)
        st.write("""The Keras Python library makes creating deep learning models fast and easy.
        The sequential API allows  to create models layer-by-layer for most problems. It is limited in that it does not allow to create models that share layers or have multiple inputs or outputs.
        The functional API in Keras is an alternate way of creating models that offers a lot more flexibility, including creating more complex models.""")
        # AlgoSequentiel(x_train , x_test ,  y_train , y_test , y_test1 , y1 )

