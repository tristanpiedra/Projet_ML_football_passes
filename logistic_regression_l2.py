import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from Scores import *
from create_dataframe import * 
from create_dataframe_regression import * 
from numpy.random import *
import time 
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



def regression_logistique_l2 (nbiter, train_proportion):

    data = pd.read_csv("dataframe_regression_centre.csv")
    data  = data.sort_values(["passe_id", "receveur_potentiel"]).reset_index().drop(["index"], axis = 1)
    data.drop(["Unnamed: 0"], axis = 1)

    col = data.columns.tolist()
    col = col[1:]
    data = data[col]
    data = data.drop(["sender_id"], 1)
    data = data.drop(["premiere_distance_sender"], 1)
    data= data.drop(["seconde_distance_sender"], 1)


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    #scaler = StandardScaler()
    method = LogisticRegression(penalty = "l2", C = 0.1)

    n_passes = 10039
    
    matrice_coef = np.zeros((nbiter, 9))
    liste_scores = np.zeros(nbiter)
    
    
    
    for niter in range (nbiter):
        
        liste_passes_train = choice(range(1,n_passes+1), size = int(train_proportion * n_passes), replace = False)
        liste_passes_train.sort()
        

        n_passes_train = int(train_proportion * n_passes) 
        n_passes_test = n_passes - n_passes_train
        #data = scaler.fit_transform (data) 
        X_train = data [ data["passe_id"].isin(liste_passes_train)]
        y_train = data [ data["passe_id"].isin(liste_passes_train)] ["passe"]
        X_test = data [ ~data["passe_id"].isin(liste_passes_train)]
        y_test = data [ ~data["passe_id"].isin(liste_passes_train)] ["passe"]
        #X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=37)


        vect_receveur_potentiel = X_test["receveur_potentiel"]
        vect_vrai_receveur = X_test ["receiver_id"]

        X_train = X_train.drop(["passe"], 1)
        X_train = X_train.drop(["passe_id"], 1)
        X_train = X_train.drop(["receveur_potentiel"], 1)
        X_train = X_train.drop(["receiver_id"], 1)

        X_test = X_test.drop(["passe"], 1)
        X_test = X_test.drop(["passe_id"], 1)
        X_test = X_test.drop(["receveur_potentiel"], 1)
        X_test = X_test.drop(["receiver_id"], 1)


        method = method.fit(X_train, y_train)
        proba = method.predict_proba (X_test)
        pred = method.predict (X_test)
        score = method.score(X_test, y_test)
        coef = method.coef_
        
        matrice_coef [niter,:] = coef

        result = proba[:,1]

        #on recupere dans prediction_indice les indices des lignes du dataframe test ou il y a la proba max 
        prediction_indice = np.zeros(n_passes_test)
        for i in range(n_passes_test):
            prediction_indice[i] = int(np.argmax(result[i*14 : (i+1) * 14 ]) + (i*14))


        X_test["receveur_potentiel"] = vect_receveur_potentiel 


        #on recupere dans prediction les receveurs potentiels qui ont le plus de chance de recevoir la passe  
        prediction = np.zeros(n_passes_test)
        count = 0
        for i in prediction_indice:
            prediction[count] = X_test.iloc[int(i)]["receveur_potentiel"] 
            count += 1


        verif = np.zeros(len(prediction))
        for i in range(len(prediction)):
            verif [i] = np.array(vect_vrai_receveur) [i*14]


        taux_reussite = np.mean ((verif - prediction)%14 == 0)
        
        liste_scores[niter] = taux_reussite
        
        moyenne_matrice_coef = np.zeros(9)
        for i in range(9):
            moyenne_matrice_coef[i] = np.mean(matrice_coef[:,i])
        
    return liste_scores, np.mean(liste_scores), matrice_coef, moyenne_matrice_coef, proba

def regression_logistique_L2_OverSamp (nbiter, train_proportion, centered=False):
    if centered==False:
        data = pd.read_csv("dataframe_regression.csv")
    else:
        data = pd.read_csv("dataframe_regression_centre.csv")
    data  = data.sort_values(["passe_id", "receveur_potentiel"]).reset_index().drop(["index"], axis = 1)
    data.drop(["Unnamed: 0"], axis = 1)

    col = data.columns.tolist()
    col = col[1:]
    data = data[col]
    data = data.drop(["sender_id"], 1)
    data = data.drop(["premiere_distance_sender"], 1)
    data= data.drop(["seconde_distance_sender"], 1)


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    #scaler = StandardScaler()
    method = LogisticRegression(penalty = "l2", C = 0.1)
    
    n_passes = 10039
    
    matrice_coef = np.zeros((nbiter, 9))
    liste_scores = np.zeros(nbiter)
    
    table=[]
    
    
    
    for niter in range (nbiter):
        
        liste_passes_train = choice(range(1,n_passes+1), size = int(train_proportion * n_passes), replace = False)
        liste_passes_train.sort()
        

        n_passes_train = int(train_proportion * n_passes) 
        n_passes_test = n_passes - n_passes_train
        #data = scaler.fit_transform (data) 
        X_train = data [ data["passe_id"].isin(liste_passes_train)]
        y_train = data [ data["passe_id"].isin(liste_passes_train)] ["passe"]
        X_test = data [ ~data["passe_id"].isin(liste_passes_train)]
        y_test = data [ ~data["passe_id"].isin(liste_passes_train)] ["passe"]
        #X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=37)
        from imblearn.over_sampling import SMOTE, ADASYN
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        print(sorted(Counter(y_resampled).items()))
        vect_receveur_potentiel = X_test["receveur_potentiel"]
        vect_vrai_receveur = X_test ["receiver_id"]

        X_resampled = X_resampled.drop(["passe"], 1)
        X_resampled = X_resampled.drop(["passe_id"], 1)
        X_resampled = X_resampled.drop(["receveur_potentiel"], 1)
        X_resampled = X_resampled.drop(["receiver_id"], 1)

        X_test = X_test.drop(["passe"], 1)
        X_test = X_test.drop(["passe_id"], 1)
        X_test = X_test.drop(["receveur_potentiel"], 1)
        X_test = X_test.drop(["receiver_id"], 1)


        method = method.fit(X_resampled, y_resampled)
        proba = method.predict_proba (X_test)
        pred = method.predict (X_test)
        score = method.score(X_test, y_test)
        coef = method.coef_
        
        table += [pd.crosstab(pred, y_test)]
        
        matrice_coef [niter,:] = coef

        result = proba[:,1]

        #on recupere dans prediction_indice les indices des lignes du dataframe test ou il y a la proba max 
        prediction_indice = np.zeros(n_passes_test)
        for i in range(n_passes_test):
            prediction_indice[i] = int(np.argmax(result[i*14 : (i+1) * 14 ]) + (i*14))


        X_test["receveur_potentiel"] = vect_receveur_potentiel 


        #on recupere dans prediction les receveurs potentiels qui ont le plus de chance de recevoir la passe  
        prediction = np.zeros(n_passes_test)
        count = 0
        for i in prediction_indice:
            prediction[count] = X_test.iloc[int(i)]["receveur_potentiel"] 
            count += 1


        verif = np.zeros(len(prediction))
        for i in range(len(prediction)):
            verif [i] = np.array(vect_vrai_receveur) [i*14]


        taux_reussite = np.mean ((verif - prediction)%14 == 0)
        
        liste_scores[niter] = taux_reussite
        
        moyenne_matrice_coef = np.zeros(9)
        for i in range(9):
            moyenne_matrice_coef[i] = np.mean(matrice_coef[:,i])
    
    moyenne_table = (table[0] + table[1] + table[2] + table[3] + table[4] + table[5] + table[6] + table[7] + table[8] + table[9]) / 10
        
    return liste_scores, np.mean(liste_scores), matrice_coef, moyenne_matrice_coef, proba, moyenne_table
