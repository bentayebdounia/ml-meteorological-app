import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ## 1.  Préparation des données
import pandas as pd

def PreparationData():
    # 1. Lire le fichier CSV et trier les données
    df = pd.read_csv("table_data.csv")
    df["date_mesure"] = pd.to_datetime(df["date_mesure"])
    df.sort_values(by=["date_mesure", "ville"], inplace=True)
    df.reset_index(drop=True, inplace=True)  # Éviter de créer une colonne "index"

    # 2. Supprimer les colonnes inutiles
    df.drop(columns=['id', 'Annee', 'MoisMesure', 'lever_soleil', 'coucher_soleil'], inplace=True)

    # 3. Construire les tables J-365, J-30, J-2, J-1, J
    df_j0 = df.iloc[1460:].reset_index(drop=True)
    df_j1 = df.iloc[1456:].reset_index(drop=True)
    df_j2 = df.iloc[1452:].reset_index(drop=True)
    df_j30 = df.iloc[1340:].reset_index(drop=True)
    
    # 4. Fusionner les tables avec des suffixes explicites
    data_final = df.merge(df_j30, left_index=True, right_index=True, suffixes=('_j-365', '_j-30'))
    data_final = data_final.merge(df_j2, left_index=True, right_index=True, suffixes=('', '_j-2'))
    data_final = data_final.merge(df_j1, left_index=True, right_index=True, suffixes=('', '_j-1'))
    data_final = data_final.merge(df_j0, left_index=True, right_index=True, suffixes=('', '_j'))
    print(data_final.dtypes)
   
    for col in data_final.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        data_final[col] = encoder.fit_transform(data_final[col])

    
    for col in data_final.select_dtypes(include=['datetime64']).columns:
        data_final[col] = data_final[col].astype('int64') // 10**9
    

    # 5. Supprimer les colonnes "index_x" et "index_y" s'ils existent
    data_final.drop(columns=[col for col in ['index_x', 'index_y'] if col in data_final.columns], inplace=True)

    # 6. Définir X (features) et Y (target)
    x = [col for col in data_final.columns if '_j' in col]  # Sélectionner toutes les colonnes avec "_j"
    y = ['ville', 'date_mesure', 'Temperature_maximale°', 'Temperature_minimale°',
         'Vitesse_de_vent_km_h', 'Temperatur_de_vent°', 'Precipitation_mm',
         'Humidite_en_pourcentage', 'Visibilite_km',
         'Couverture_nuageuse_en_pourcentage', 'indice_chaleur', 'Point_rosee_°C']

    X = data_final[x]
    Y = data_final[y]

    return data_final, X, Y
 

# #### 70% apprentissage et 30% test :
def TrainingAndTestDATA (X,Y):

    x_train , x_test , y_train1 , y_test1 = train_test_split( X , Y , test_size=0.3 )
    y1 = ['Temperature_maximale°', 'Temperature_minimale°',
        'Vitesse_de_vent_km_h', 'Temperatur_de_vent°', 'Precipitation_mm',
        'Humidite_en_pourcentage', 'Visibilite_km','Couverture_nuageuse_en_pourcentage', 'indice_chaleur','Point_rosee_°C' ]

    
    y_train = y_train1 [y1]
    y_test = y_test1 [y1]

    return x_train , x_test ,  y_train , y_test , y_test1 , y1


