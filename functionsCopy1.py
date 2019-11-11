import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ScoresCopy1 import *


#On calcule la distance entre deux joueurs quelconques
def distance (dfligne, joueur_un, joueur_deux) : 
    d = np.sqrt ((dfligne["x_{}".format(int(joueur_un))] - dfligne["x_{}".format(int(joueur_deux))])**2 + (dfligne["y_{}".format(int(joueur_un))] - dfligne["y_{}".format(int(joueur_deux))])**2)
    return d

#fonction qui renvoie le décalage nécessaire pour balayer l'équipe adverse du passeur 
def shift_equipe_adverse (sender) :
    if sender < 15:
        return 14
    else:
        return 0

#fonction qui renvoie le décalage nécessaire pour balayer l'équipe du passeur 
def shift_equipe_partenaire (sender) :
    if sender < 15:
        return 0
    else:
        return 14


#On regarde si un adversaire est dans le périmètre ou pas 
def perimetre (dfligne, sender, receveur) : 
    Trouve = False
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        if distance(dfligne, receveur, i) < 700 :
            Trouve = True 
            return Trouve
    return Trouve 

#On regarde combien d'adversaires sont dans le périmètre  
def nombre_adversaires (dfligne, sender, receveur) : 
    nombre_adversaires = 0
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        if distance(dfligne, receveur, i) < 700:
            nombre_adversaires += 1
    return nombre_adversaires


#Systeme de passe backward/Forward classique 

def DirectionPasse (dfligne) :
    sender = dfligne["sender_id"]
    receiver = dfligne["receiver_id"]
    SenderX = dfligne["x_{}".format(int(sender))] 
    ReceiverX = dfligne["x_{}".format(int(receiver))]
    if sender < 15 :  #equipe a droite 
        if SenderX < ReceiverX :
            Direction = "Backward"
        else:
            Direction = "Forward"
    else:  #equipe a gauche
        if SenderX > ReceiverX:
            Direction = "Backward"
        else:
            Direction = "Forward"
    return Direction

#Systeme de passe avec ajout de passes laterales

def DirectionPasse_amelioree (dfligne) :
    sender = dfligne["sender_id"]
    receiver = dfligne["receiver_id"]
    SenderX, SenderY = dfligne ["x_{}".format(sender)], dfligne ["y_{}".format(sender)]
    ReceiverX, ReceiverY = dfligne["x_{}".format(receiver)] , dfligne ["y_{}".format(receiver)]
    if sender < 15:  #equipe a droite 
        if np.abs( - SenderX + ReceiverX) > np.abs(ReceiverY - SenderY) :    #passe non laterale
            if SenderX < ReceiverX :
                Direction = "Backward"
            else :
                Direction = "Forward"
        else :
            Direction = "Sideways"
    else :  #equipe a gauche
        if np.abs(- SenderX + ReceiverX) > np.abs(ReceiverY - SenderY) :
            if SenderX > ReceiverX :
                Direction = "Backward"
            else :
                Direction = "Forward"
        else :
            Direction = "Sideways"
    return Direction





#creation de la matrice de score pour chaque coéquipier du passeur 

def matrice_de_prediction (pred, df) :
    NbLignes = df.shape [0]
    ScoreTeamates = np.zeros ((NbLignes, 14))
    for i in range (NbLignes) :
        dfligne=df.iloc[i]
        sender = int(dfligne ["sender_id"])
        for j in range (1 + shift_equipe_partenaire(sender), 15 + shift_equipe_partenaire(sender)) :
                if j  == sender:
                    ScoreTeamates [i, (j % 14) - 1] = 1000000   #on fausse le score du passeur
                else:
                    ScoreTeamates [i, (j % 14) - 1] = pred (dfligne, sender, j)
    return ScoreTeamates




#renvoie un vecteur avec toutes les predictions pour une matrice de scores donnée 


def prediction (mat, df) :   #applique a toutes les lignes 
    mat = pd.DataFrame(mat)
    sender = df["sender_id"]
    mat["sender"] = sender
    prediction = mat.apply(lambda x: np.argmin(x[:-1]) + shift_equipe_partenaire(x["sender"]) + 1 , axis = 1)
    return prediction


#création variable interception
def ajout_interception (df) :
    nblignes = df.shape [0]
    interception = np.zeros(nblignes)
    for i in range(nblignes) :
        dfligne=df.iloc[i]
        if ((int(dfligne["sender_id"]) < 15) and (int(dfligne["receiver_id"]) > 14)):    #sender equipe 1
            interception [i] = 1
        if ((int(dfligne["sender_id"]) > 14) and (int(dfligne["receiver_id"]) < 15)):    #sender equipe 2
            interception [i] = 1
    df ["interception"] = interception
    return df


#création variable distance de passe
'''def ajout_dist_passe (df) :
    NbLignes = df.shape[0]
    distpasse = np.zeros(NbLignes)
    for i in range(NbLignes):
        dfligne=df.iloc[i]
        sender = dfligne["sender_id"]
        receveur = dfligne["receiver_id"]
        distpasse [i] = distance (dfligne , sender, receveur)
    df["DistPasse"] = distpasse
    return df '''

#création variable temps de passe
'''def ajout_temps_passe (df) : 
    df["temps_passe"] = df["time_end"] - df["time_start"]
    return df '''

#création variable team side
def ajout_team_side (df) : 
    NbLignes = df.shape [0]
    senderteamside = ["" for x in range(NbLignes)]
    for i in range(NbLignes):
        dfligne=df.iloc[i]
        if int(dfligne["sender_id"])>14 :
            senderteamside [i] = "Left"
        else:
            senderteamside [i] = "Right"
    df ["SenderTeamSide"] = senderteamside 
    return df

#fonction de suppression interception d'un dataframe
def suppr_interception (df) :
    df = df[df['interception']==0]
    df = df.reset_index().drop(["index"], axis=1)
    return df  

#fonction qui supprime les fausses donnees (sender = receiver ou sender,receiver = nan)
def suppr_fausses_donnees (df):
    df = df[df["sender_id"]!= df ["receiver_id"]]
    df["fake"] = df.apply(lambda x: x["x_{}".format(int(x["sender_id"]))]==100000 or x["x_{}".format(int(x["receiver_id"]))]==100000, axis = 1)
    df = df[df["fake"]==False]
    return df

#CREATION DE FEATURES DE L'ARTICLE 

#fonction qui retourne les 2 plus proches adversaires du sender. on l'ajoute avec un apply dans le dataframe. 

def distances_sender_opponents (dfligne):
    sender = dfligne["sender_id"]
    liste = []
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        liste += [distance(dfligne, sender, i)]
    result1 = min(liste)
    del liste[np.argmin(liste)]
    result2 = min(liste)
    return result1, result2

#ajout dans le dataframe
def ajout_distances_sender (df):
    df["premiere_distance_sender"] = df.apply(lambda line: distances_sender_opponents(line)[0] , axis = 1)
    df["seconde_distance_sender"] = df.apply(lambda line: distances_sender_opponents(line)[1] , axis = 1)
    return df
        



def creation_dataframe (df) :
    
    #remplacement des nan
    df = df.replace(np.nan, 100000)
    
    #création variable interception
    df = ajout_interception (df)
    
    #on enlève les interceptions    
    df = suppr_interception (df)
    
    #création variable distance de passe
    #df = ajout_dist_passe (df)
    
    #creation variable temps de passe
    #df = ajout_temps_passe (df)
    
    #creation variable côté
    df = ajout_team_side (df)
    
    #suppression données aberrantes
    df = suppr_fausses_donnees (df)
    
    #ajout distances_sender 
    df = ajout_distances_sender (df)
    
    #matrices de scores
    mat_score1 = matrice_de_prediction (Score1, df)
    mat_score2 = matrice_de_prediction (Score2, df)
    mat_score3 = matrice_de_prediction (Score3, df)
    mat_score4 = matrice_de_prediction (Score4, df)
    
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
    
    
    