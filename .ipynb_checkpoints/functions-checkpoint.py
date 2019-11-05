import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Scores import *


#On calcule la distance entre deux joueurs quelconques
def distance (df, joueur_un, joueur_deux, ligne) : 
    d = np.sqrt ((df["x_{}".format(int(joueur_un))][ligne] - df["x_{}".format(int(joueur_deux))][ligne])**2 + (df["y_{}".format(int(joueur_un))][ligne] - df["y_{}".format(int(joueur_deux))][ligne])**2)
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
def perimetre (df, sender, receveur, ligne) : 
    Trouve = False
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        if distance(df, receveur, i,ligne) < 700 :
            Trouve = True 
    return Trouve 

#On regarde combien d'adversaires sont dans le périmètre  
def nombre_adversaires (df, sender, receveur, ligne) : 
    nombre_adversaires = 0
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        if distance(df, receveur, i,ligne) < 700:
            nombre_adversaires += 1
    return nombre_adversaires


#Systeme de passe backward/Forward classique 

def DirectionPasse (Ligne, df) :
    sender = df["sender_id"][Ligne]
    receiver = df["receiver_id"][Ligne]
    SenderX = df["x_{}".format(int(sender))] [Ligne]
    ReceiverX = df["x_{}".format(int(receiver))][Ligne]
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

def DirectionPasse_amelioree (Ligne, df) :
    sender = df["sender_id"][Ligne]
    receiver = df["receiver_id"][Ligne]
    SenderX, SenderY = df ["x_{}".format(sender)][Ligne], df ["y_{}".format(sender)][Ligne]
    ReceiverX, ReceiverY = df["x_{}".format(receiver)][Ligne] , df ["y_{}".format(receiver)] [Ligne]
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
        sender = df ["sender_id"][i]
        for j in range (1 + shift_equipe_partenaire(sender), 15 + shift_equipe_partenaire(sender)) :
                if j  == sender:
                    ScoreTeamates [i, (j % 14) - 1] = 1000000   #on fausse le score du passeur
                else:
                    ScoreTeamates [i, (j % 14) - 1] = pred (df, sender, j, i)
    return ScoreTeamates






#renvoie un vecteur avec toutes les predictions pour une matrice de scores donnée 

def prediction (mat, df) :
    NbLignes = df.shape [0]
    prediction = np.zeros (NbLignes)
    for i in range (NbLignes) :
        if df ["sender_id"][i] < 15 : #equipe1
            prediction [i] = np.argmin (mat[i, :]) + 1
        else:
            prediction [i] = np.argmin (mat[i, :]) + 15
    return prediction



#création variable interception
def ajout_interception (df) :
    nblignes = df.shape [0]
    interception = np.zeros(nblignes)
    for i in range(len(df["x_1"])) :
        if ((df["sender_id"] [i] < 15) and (df["receiver_id"] [i] > 14)):    #sender equipe 1
            interception [i] = 1
        if ((df["sender_id"] [i] > 14) and (df["receiver_id"][i] < 15)):    #sender equipe 2
            interception [i] = 1
    df ["interception"] = interception
    return df


#création variable distance de passe
def ajout_dist_passe (df) :
    NbLignes = df.shape[0]
    distpasse = np.zeros(len(df["x_1"]))
    for i in range(len(df["x_1"])):
        sender = df ["sender_id"][i]
        receveur = df ["receiver_id"][i]
        distpasse [i] = distance (df , sender, receveur, i)
    df["DistPasse"] = distpasse
    return df 

#création variable temps de passe
def ajout_temps_passe (df) : 
    df["temps_passe"] = df["time_end"] - df["time_start"]
    return df

#création variable team side
def ajout_team_side (df) : 
    NbLignes = df.shape [0]
    senderteamside = ["" for x in range(NbLignes)]
    for i in range(NbLignes):
        if df ["sender_id"][i] >14 :
            senderteamside [i] = "Left"
        else:
            senderteamside [i] = "Right"
    df ["SenderTeamSide"] = senderteamside 
    return df

#fonctions de suppression interception d'un dataframe
def suppr_interception (df) :
    df = df[df['interception']==0]
    return df

def creation_dataframe (df) :
    
    #remplacement des nan
    df = df.replace(np.nan, 100000)
    
    #création variable interception
    df = ajout_interception (df)
    
    #on enlève les interceptions    
    df = suppr_interception (df)
    
    #création variable distance de passe
    df = ajout_dist_passe (df)
    
    #creation variable temps de passe
    df = ajout_temps_passe (df)
    
    #creation variable côté
    df = ajout_team_side (df)
    
    #matrices de scores
    mat_score1 = matrice_de_prediction (Score1, df)
    mat_score2 = matrice_de_prediction (Score2, df)
    mat_score3 = matrice_de_prediction (Score3, df)
    mat_score4 = matrice_de_prediction (Score4, df)
    
    #Variable de prediction score 1
    df['DistTeamate1']=mat_score1[:,0]
    df['DistTeamate2']=mat_score1[:,1]
    df['DistTeamate3']=mat_score1[:,2]
    df['DistTeamate4']=mat_score1[:,3]
    df['DistTeamate5']=mat_score1[:,4]
    df['DistTeamate6']=mat_score1[:,5]
    df['DistTeamate7']=mat_score1[:,6]
    df['DistTeamate8']=mat_score1[:,7]
    df['DistTeamate9']=mat_score1[:,8]
    df['DistTeamate10']=mat_score1[:,9]
    df['DistTeamate11']=mat_score1[:,10]
    df['DistTeamate12']=mat_score1[:,11]
    df['DistTeamate13']=mat_score1[:,12]
    df['DistTeamate14']=mat_score1[:,13]
    #Variable de prediction score 2
    df['Score2Teamate1']=mat_score2[:,0]
    df['Score2Teamate2']=mat_score2[:,1]
    df['Score2Teamate3']=mat_score2[:,2]
    df['Score2Teamate4']=mat_score2[:,3]
    df['Score2Teamate5']=mat_score2[:,4]
    df['Score2Teamate6']=mat_score2[:,5]
    df['Score2Teamate7']=mat_score2[:,6]
    df['Score2Teamate8']=mat_score2[:,7]
    df['Score2Teamate9']=mat_score2[:,8]
    df['Score2Teamate10']=mat_score2[:,9]
    df['Score2Teamate11']=mat_score2[:,10]
    df['Score2Teamate12']=mat_score2[:,11]
    df['Score2Teamate13']=mat_score2[:,12]
    df['Score2Teamate14']=mat_score2[:,13]
    #Variable de prediction score 3
    
    df['Score3Teamate1']=mat_score3[:,0]
    df['Score3Teamate2']=mat_score3[:,1]
    df['Score3Teamate3']=mat_score3[:,2]
    df['Score3Teamate4']=mat_score3[:,3]
    df['Score3Teamate5']=mat_score3[:,4]
    df['Score3Teamate6']=mat_score3[:,5]
    df['Score3Teamate7']=mat_score3[:,6]
    df['Score3Teamate8']=mat_score3[:,7]
    df['Score3Teamate9']=mat_score3[:,8]
    df['Score3Teamate10']=mat_score3[:,9]
    df['Score3Teamate11']=mat_score3[:,10]
    df['Score3Teamate12']=mat_score3[:,11]
    df['Score3Teamate13']=mat_score3[:,12]
    df['Score3Teamate14']=mat_score3[:,13]
    
    #Variable de prediction score 4
    
    df['Score4Teamate1']=mat_score4[:,0]
    df['Score4Teamate2']=mat_score4[:,1]
    df['Score4Teamate3']=mat_score4[:,2]
    df['Score4Teamate4']=mat_score4[:,3]
    df['Score4Teamate5']=mat_score4[:,4]
    df['Score4Teamate6']=mat_score4[:,5]
    df['Score4Teamate7']=mat_score4[:,6]
    df['Score4Teamate8']=mat_score4[:,7]
    df['Score4Teamate9']=mat_score4[:,8]
    df['Score4Teamate10']=mat_score4[:,9]
    df['Score4Teamate11']=mat_score4[:,10]
    df['Score4Teamate12']=mat_score4[:,11]
    df['Score4Teamate13']=mat_score4[:,12]
    df['Score4Teamate14']=mat_score4[:,13]
    
    #creation variables qui affichent la prédiction
    df["predic1"] = prediction (mat_score1, df)
    df["predic2"] = prediction (mat_score2, df)
    df["predic3"] = prediction (mat_score3, df)
    df["predic4"] = prediction (mat_score4, df)
    
    return df
    
    
    