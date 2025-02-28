import streamlit as st
import pandas as pd
from sklearn import *

def Predict ():

    date = st.date_input("Pick a date that you want to predict : ")
    date_month =date.month

    if ( date.day ==1 or date.day == 2):
        if ( date.month == 2 or date.month == 4 or date.month ==6 or date.month == 8 or date.month == 9 or date.month == 11 or date.month == 1) :
                date_day_1 = 31
                date_day_2 = 30
                
                if (date.month == 1) : date_month = 12
                else : date_month = date.month - 1
        elif (date.month == 3 ) :
            date_day_1 = 28
            date_day_2 = 27
            date_month = date.month-1
        else :
                date_day_1 = 30
                date_day_2 = 29
                date_month = date.month-1
    else :
        date_day_1 = date.day-1
        date_day_2 = date.day-2
    if (date.month == 1) : date_day_30 = 12
    else : date_day_30 = date.month - 1
    st.selectbox("Select city :" ,("Oran" , "Mostaganem" , "Sidi bel Abbes" , "Tlamcen"))
    col1 , col2 , col3 , col4 = st.columns(4)

    with st.form(key='table'):
        with col1:
            
            st.write('#### Enter the date settings of :')
            st.write(date_day_1 , "/" , date_month , "/" , date.year)

            max_tempD1=st.number_input(label='Maximum temperature D-1:', value = 0 , max_value=50 , min_value=-10 )
            
            min_tempD1=st.number_input(label='Minimum temperature D-1:' , value = 0, max_value=50 , min_value=-10)
        
            wind_spD1=st.number_input(label='Wind speed D-1:', value = 0 , max_value=100 , min_value=0)
            
            wind_tempD1=st.number_input(label='Wind temperature D-1:', value = 0 , max_value=50 , min_value=-10)
            
            precD1=st.number_input(label='Precipitation D-1:', value = 0 , max_value=1000 , min_value=0)
            
            humD1=st.number_input(label='Humidity D-1:', value = 0 , max_value=100 , min_value=0)
        
            visD1=st.number_input(label='Visibility D-1:', value = 0 , max_value=100 , min_value=0)
            
            Cloud_cvrD1=st.number_input(label='Cloud cover D-1:', value = 0 , max_value=100 , min_value=0)
            
            Heart_idxD1=st.number_input(label='Heat index D-1:', value = 0 , max_value=50 , min_value=-10)
            
            dew_ptD1=st.number_input(label='Dew point D-1:', value = 0 , max_value=50 , min_value=-10)


        with col2:
            st.write('#### Enter the date settings of: ')
            st.write(date_day_2 , "/" , date_month , "/" , date.year)

            max_tempD2=st.number_input(label='Maximum temperature D-2:', value = 0 , max_value=50 , min_value=-10 )
            
            min_tempD2=st.number_input(label='Minimum temperature D-2:' , value = 0, max_value=50 , min_value=-10)
        
            wind_spD2=st.number_input(label='Wind speed D-2:', value = 0 , max_value=100 , min_value=0)
            
            wind_tempD2=st.number_input(label='Wind temperature D-2:', value = 0 , max_value=50 , min_value=-10)
            
            precD2=st.number_input(label='Precipitation D-2:', value = 0 , max_value=1000 , min_value=0)
            
            humD2=st.number_input(label='Humidity D-2:', value = 0 , max_value=100 , min_value=0)
        
            visD2=st.number_input(label='Visibility D-2:', value = 0 , max_value=100 , min_value=0)
            
            Cloud_cvrD2=st.number_input(label='Cloud cover D-2:', value = 0 , max_value=100 , min_value=0)
            
            Heart_idxD2=st.number_input(label='Heat index D-2:', value = 0 , max_value=50 , min_value=-10)
            
            dew_ptD2=st.number_input(label='Dew point D-2:', value = 0 , max_value=50 , min_value=-10)
                    

        with col3:
            st.write('#### Enter the date settings of :')
            st.write(date.day , "/" , date_day_30 , "/" , date.year)
            
            max_tempM1=st.number_input(label='Maximum temperature M-1:', value = 0 , max_value=50 , min_value=-10 )
            
            min_tempM1=st.number_input(label='Minimum temperature M-1:' , value = 0, max_value=50 , min_value=-10)
        
            wind_spM1=st.number_input(label='Wind speed M-1:', value = 0 , max_value=100 , min_value=0)
            
            wind_tempM1=st.number_input(label='Wind temperature M-1:', value = 0 , max_value=50 , min_value=-10)
            
            precM1=st.number_input(label='Precipitation M-1:', value = 0 , max_value=1000 , min_value=0)
            
            humM1=st.number_input(label='Humidity M-1:', value = 0 , max_value=100 , min_value=0)
        
            visM1=st.number_input(label='Visibility M-1:', value = 0 , max_value=100 , min_value=0)
            
            Cloud_cvrM1=st.number_input(label='Cloud cover M-1:', value = 0 , max_value=100 , min_value=0)
            
            Heart_idxM1=st.number_input(label='Heat index M-1:', value = 0 , max_value=50 , min_value=-10)
            
            dew_ptM1=st.number_input(label='Dew point M-1:', value = 0 , max_value=50 , min_value=-10)

        with col4:
            st.write('#### Enter the date settings of :')
            st.write(date.day , "/" , date.month , "/" , date.year - 1)
            
            max_tempY1=st.number_input(label='Maximum temperature Y-1:', value = 0 , max_value=50 , min_value=-10 )
            
            min_tempY1=st.number_input(label='Minimum temperature Y-1:' , value = 0, max_value=50 , min_value=-10)
        
            wind_spY1=st.number_input(label='Wind speed Y-1:', value = 0 , max_value=100 , min_value=0)
            
            wind_tempY1=st.number_input(label='Wind temperature Y-1:', value = 0 , max_value=50 , min_value=-10)
            
            precY1=st.number_input(label='Precipitation Y-1:', value = 0 , max_value=1000 , min_value=0)
            
            humY1=st.number_input(label='Humidity Y-1:', value = 0 , max_value=100 , min_value=0)
        
            visY1=st.number_input(label='Visibility Y-1:', value = 0 , max_value=100 , min_value=0)
            
            Cloud_cvrY1=st.number_input(label='Cloud cover Y-1:', value = 0 , max_value=100 , min_value=0)
            
            Heart_idxY1=st.number_input(label='Heat index Y-1:', value = 0 , max_value=50 , min_value=-10)
            
            dew_ptY1=st.number_input(label='Dew point Y-1:', value = 0 , max_value=50 , min_value=-10)
    
        pred_button = st.form_submit_button(label='Predict')

        if pred_button:
            x_pred=pd.DataFrame({
            'Temperature_maximale°_j365' : [max_tempY1],
            'Temperature_minimale°_j365' : [min_tempY1], 
            'Vitesse_de_vent_km_h_j365' : [wind_spY1],
            'Temperatur_de_vent°_j365' : [wind_tempY1], 
            'Precipitation_mm_j365' : [precY1],
            'Humidite_en_pourcentage_j365' : [humY1], 
            'Visibilite_km_j365' : [visY1],
            'Couverture_nuageuse_en_pourcentage_j365' : [Cloud_cvrY1], 
            'indice_chaleur_j365' : [Heart_idxY1],
            'Point_rosee_°C_j365' : [dew_ptY1] ,
            'Temperature_maximale°_j30' : [max_tempM1],
            'Temperature_minimale°_j30' : [min_tempM1], 
            'Vitesse_de_vent_km_h_j30' : [wind_spM1],
            'Temperatur_de_vent°_j30' : [wind_tempM1], 
            'Precipitation_mm_j30' : [precM1],
            'Humidite_en_pourcentage_j30' : [humM1], 
            'Visibilite_km_j30' : [visM1],
            'Couverture_nuageuse_en_pourcentage_j30' : [Cloud_cvrM1], 
            'indice_chaleur_j30' : [Heart_idxM1],
            'Point_rosee_°C_j30'  : [dew_ptM1],
            'Temperature_maximale°_x' : [max_tempD2], 
            'Temperature_minimale°_x' : [ min_tempD2],
            'Vitesse_de_vent_km_h_x' : [wind_spD2], 
            'Temperatur_de_vent°_x' : [wind_tempD2], 
            'Precipitation_mm_x' : [precD2],
            'Humidite_en_pourcentage_x' : [humD2], 
            'Visibilite_km_x' : [visD2],
            'Couverture_nuageuse_en_pourcentage_x' : [Cloud_cvrD2], 
            'indice_chaleur_x' : [Heart_idxD2],
            'Point_rosee_°C_x' : [dew_ptD2], 
            'Temperature_maximale°_y': [max_tempD1],
            'Temperature_minimale°_y': [min_tempD1],
            'Vitesse_de_vent_km_h_y' : [wind_spD1 ],
            'Temperatur_de_vent°_y' : [wind_tempD1],
            'Precipitation_mm_y' :[precD1],
            'Humidite_en_pourcentage_y' : [humD1],
            'Visibilite_km_y' : [visD1],
            'Couverture_nuageuse_en_pourcentage_y' : [Cloud_cvrD1],
            'indice_chaleur_y' : [Heart_idxD1],
            'Point_rosee_°C_y' :[dew_ptD1]
            
            })
            #st.write(x_pred)
            #Result_predict = model_pred.predict(x_pred)
            #y_pred_XGBR = modelPredict.predict(x_test)
            #st.write(Result_predict )
            return x_pred