import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Scores import *
from functions import *

def creation_dataframe_regression (df) :
    dataframe = []
    for i in range (1, 15):
        dataframe.append(df[['sender_id', 'receiver_id', 'Score1Teamate_{}'.format(i), 'Score2Teamate_{}'.format(i), 'Score3Teamate_{}'.format(i), 'Score4Teamate_{}'.format(i), 'premiere_distances_receveur_{}'.format(i), 'seconde_distances_receveur_{}'.format(i), 'premiere_distance_sender', 'seconde_distance_sender', 'distance_ligne_passe_{}'.format(i), 'AdversaireDansCone_{}'.format(i)]])
        dataframe[i-1] = dataframe[i-1].rename (columns = {"Score1Teamate_{}".format(i) : "score1_distance", "Score2Teamate_{}".format(i) : "score2", "Score3Teamate_{}".format(i) : "score3", "Score4Teamate_{}".format(i) : "score4", "premiere_distances_receveur_{}".format(i) : "premiere_distance_receveur", "seconde_distances_receveur_{}".format(i) : "seconde_distance_receveur", 'distance_ligne_passe_{}'.format(i) : 'distance_ligne_passe', 'AdversaireDansCone_{}'.format(i) : 'adversaire_dans_cone'})

    df = pd.concat (dataframe, axis = 0)
    df = df.reset_index().drop(["index"], axis=1)
    df ["receveur_potentiel"] = (np.floor(np.array(df.index.values / 10039)) + 1).astype(int)
    col = df.columns.tolist()
    col = col[0:2] + [col[-1]] + col[2:-1]
    df = df[col]
    df ["passe"] = (df["receveur_potentiel"] == (df["receiver_id"]) % 14).astype(int)
    df ["adversaire_dans_cone"] = df ["adversaire_dans_cone"].astype(int)
    x = np.arange(10039) + 1
    x = np.tile(x,14)
    df["passe_id"] = x
    col = df.columns.tolist()
    col = col[0:3] + col[-2:] + col[3:-2]
    df = df[col]
    
    return df