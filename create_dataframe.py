import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ScoresCopy1 import *
from functionsCopy1 import *




def creation_dataframe (df) :
    
    #remplacement des nan
    df = df.replace(np.nan, 100000)
    
    #création variable interception
    df = ajout_interception (df)
    
    #on enlève les interceptions    
    df = suppr_interception (df)
    
    #creation variable côté
    df = ajout_team_side (df)
    
    #suppression données aberrantes
    df = suppr_fausses_donnees (df)
    
    #ajout distances_sender 
    df = ajout_distances_sender (df)
    
    #ajout distances_receveur 
    df = ajout_distances_receveur (df)
    
    
    #matrices de scores
    mat_score1 = matrice_de_prediction (Score1, df)
    print("mat1 ok")
    mat_score2 = matrice_de_prediction (Score2, df)
    print("mat2 ok")
    mat_score3 = matrice_de_prediction (Score3, df)
    print("mat3 ok")
    mat_score4 = matrice_de_prediction (Score4, df)
    print("mat4 ok")
    
    #Variables de prediction score 1
    for i in range(14):
        df['Score1Teamate_{}'.format(i+1)] = mat_score1[:,i]
        df['Score2Teamate_{}'.format(i+1)] = mat_score2[:,i]
        df['Score3Teamate_{}'.format(i+1)] = mat_score3[:,i]
        df['Score4Teamate_{}'.format(i+1)] = mat_score4[:,i]
    
    #creation variables qui affichent la prédiction
    df["predic1"] = prediction (mat_score1, df)
    df["predic2"] = prediction (mat_score2, df)
    df["predic3"] = prediction (mat_score3, df)
    df["predic4"] = prediction (mat_score4, df)
    
    return df