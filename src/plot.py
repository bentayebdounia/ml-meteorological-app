import pandas as pd
import matplotlib.pyplot as plt

def PLOTS(y_test1 , y_pred , y1 ):

    y_test1.reset_index(inplace = True)
    y_test1 = y_test1.drop([ 'index'  ],axis='columns')
    
    yPred = pd.DataFrame(y_pred, columns=y1)
    Data = pd.merge( y_test1 ,yPred  , right_index=True , left_index=True , suffixes=('_test','_pred') )
    Data.sort_values(by=["date_mesure"],inplace=True) 
    
    sizeX=28
    sizeY =12

    fig_Temp_max=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Temperature_maximale°_test'] , label = 'Actual maximum temperature  ')
    plt.plot(Data['date_mesure'],Data['Temperature_maximale°_pred'] , label = 'Predicted maximum temperature ')
    #plt.title ("compariason entre la température maximale réel et la température maximale prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Temperature on C° ")
    plt.legend()

    fig_Temp_min=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Temperature_minimale°_test'] , label = 'Actual minimum temperature')
    plt.plot(Data['date_mesure'],Data['Temperature_minimale°_pred'] , label = 'Predicted minimum temperature')
    #plt.title ("compariason entre la température minimale réel et la température minimale prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Temperature C° ")
    plt.legend()

    fig_Vitesse_vent=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Vitesse_de_vent_km_h_test'] , label = 'Actual wind speed')
    plt.plot(Data['date_mesure'],Data['Vitesse_de_vent_km_h_pred'] , label = 'Predicted wind speed')
    #plt.title ("compariason entre la vitesse de vent de vent réel et la vitesse de vent prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Wind speed on km/h ")
    plt.legend()

    fig_Temp_vent=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Temperatur_de_vent°_test'] , label = 'Actual wind temperature ')
    plt.plot(Data['date_mesure'],Data['Temperatur_de_vent°_pred'] , label = 'Predicted wind temperature ')
    #plt.title ("compariason entre la température de vent réel et la température de vent prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Temperature ° ")
    plt.legend()

    fig_Precipitation=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Precipitation_mm_test'] , label = 'Actual precipitation')
    plt.plot(Data['date_mesure'],Data['Precipitation_mm_pred'] , label = 'Predicted precipitation')
    #plt.title ("compariason entre la precipitation réel et la precipitation prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Precipitation on mm ")
    plt.legend()


    fig_Humidite=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Humidite_en_pourcentage_test'] , label = 'Actual humidity')
    plt.plot(Data['date_mesure'],Data['Humidite_en_pourcentage_pred'] , label = 'Predicted humidity')
    #plt.title ("compariason entre la Humidite réel et la Humidite  prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Humidity on % ")
    plt.legend()

    fig_Visibilite=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Visibilite_km_test'] , label = 'Actual visibility')
    plt.plot(Data['date_mesure'],Data['Visibilite_km_pred'] , label = 'Predicted visibilite_km_predit')
    #plt.title ("compariason entre la Visibilite réel et la Visibilite prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Visibility on km ")
    plt.legend()

    fig_CouverNuag=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Couverture_nuageuse_en_pourcentage_test'] , label = 'Actual cloud cover')
    plt.plot(Data['date_mesure'],Data['Couverture_nuageuse_en_pourcentage_pred'] , label = 'Predicted cloud cover')
    #plt.title ("compariason entre la Couverture nuageuse  réel et la Couverture nuageuse prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Cloud cover on %")
    plt.legend()

    fig_indice_chaleur=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['indice_chaleur_test'] , label = 'Actual heat index')
    plt.plot(Data['date_mesure'],Data['indice_chaleur_pred'] , label = 'Predited heat index')
    #plt.title ("compariason entre l'indice de chaleurréel et l'indice de chaleur prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Heat index ")
    plt.legend()

    fig_PointRos=plt.figure(figsize=(sizeX,sizeY))
    plt.plot(Data['date_mesure'],Data['Point_rosee_°C_test'] , label = 'Actual dew point')
    plt.plot(Data['date_mesure'],Data['Point_rosee_°C_pred'] , label = 'Predited dew point')
    #plt.title ("compariason entre Point rosee réel et Point rosee prédit")
    plt.xlabel("Measurement date")
    plt.ylabel("Dew point on °C ")
    plt.legend()

    
    return fig_Temp_max , fig_Temp_min , fig_Vitesse_vent , fig_Temp_vent , fig_Precipitation , fig_Humidite , fig_Visibilite , fig_CouverNuag , fig_indice_chaleur , fig_PointRos

