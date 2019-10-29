import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#On calcule pour la ligne du dataframe la distance entre sender et receveur

def distance(df, sender, receveur, ligne) : 
    d = np.sqrt((df.iloc[ligne,3+sender]-df.iloc[ligne,3+receveur])**2 + (df.iloc[ligne,31+sender]-df.iloc[ligne,31+receveur])**2)
    return d

#calcul du 2eme score

def Score2(df, sender, receveur, ligne) : 
    score = distance(df, df.iloc[ligne,2], receveur,ligne)
    Trouve = False
    if sender < 15:
        shift = 14
    else:
        shift = 0
        
    for i in range (1+shift,15+shift):
        if distance(df, receveur, i,ligne) < 700:
            Trouve = True 
            break
            
    if Trouve:
        score += 900
    return score

#calcul du 3eme score

def Score3(df, sender, receveur, ligne) : 
    score = distance(df, df.iloc[ligne,2], receveur,ligne)
    M=0
    if sender < 15:
        for i in range (15,29):
            if distance(df, receveur, i,ligne) < 700:
                M+=1
                
        
    if sender > 14:
        for i in range(1,15):
            if distance(df, receveur, i,ligne) < 700:
                M+=1
    if M >= 1:
        score += 900
    if M >= 2:
        score += 55     
    return score

#Systeme de passe backward/Forward classique 

def DirectionPasse2(Ligne, df):
    sender=df.iloc[Ligne,2]
    receiver=df.iloc[Ligne,3]
    SenderX,SenderY=df.iloc[Ligne,3+sender],df.iloc[Ligne,31+sender]
    ReceiverX,ReceiverY=df.iloc[Ligne,3+receiver],df.iloc[Ligne,31+receiver]
    
    if sender<15:  #equipe a droite 
            #passe non laterale
        if SenderX<ReceiverX:
            Direction="Backward"
        else:
            Direction="Forward"
    
            
    else:  #equipe a gauche
        
        if SenderX>ReceiverX:
            Direction="Backward"
        else:
            Direction="Forward"
        
    return Direction

#Systeme de passe avec ajout de passes laterales

def DirectionPasse(Ligne, df):
    sender=df.iloc[Ligne,2]
    receiver=df.iloc[Ligne,3]
    SenderX,SenderY=df.iloc[Ligne,3+sender],df.iloc[Ligne,31+sender]
    ReceiverX,ReceiverY=df.iloc[Ligne,3+receiver],df.iloc[Ligne,31+receiver]
    
    if sender<15:  #equipe a droite 
        if np.abs(-SenderX+ReceiverX)>np.abs(ReceiverY-SenderY):    #passe non laterale
            if SenderX<ReceiverX:
                Direction="Backward"
            else:
                Direction="Forward"
        else:
            Direction="Sideways"
    else:  #equipe a gauche
        if np.abs(-SenderX+ReceiverX)>np.abs(ReceiverY-SenderY):
            if SenderX>ReceiverX:
                Direction="Backward"
            else:
                Direction="Forward"
        else:
            Direction="Sideways"
    return Direction

#calcul du 4eme score

def Score4(df, sender, receveur, ligne) : 
    SenderX = df.iloc[ligne,3+sender]
    ReceiverX = df.iloc[ligne,3+receveur]
    score = distance(df, df.iloc[ligne,2], receveur,ligne)
    M=0
    if sender < 15:    #equipe a droite
        for i in range (15,29):
            if distance(df, receveur, i,ligne) < 700:
                M+=1
                
        
    if sender > 14:     #equipe a gauche
        for i in range(1,15):
            if distance(df, receveur, i,ligne) < 700:
                M+=1
    if M >= 1:
        score += 900
    if M >= 2:
        score += 55  
        
    if DirectionPasse2(ligne, df)=="Backward":
        score += 0.1 * np.abs(SenderX - ReceiverX)
        
    else :
        score -= 0.3 * np.abs(SenderX - ReceiverX)
        
    return score

#creation de la matrice de score pour chaque coéquipier du passeur 

def matrice_de_prediction (pred, df):
    NbLignes = df.shape[0]
    ScoreTeamates = np.zeros((NbLignes,14))
    for i in range (NbLignes):
        sender = df.iloc[i,2]
        if sender < 15:       #equipe 1
            for j in range(1,15):
                if j  == sender:
                    ScoreTeamates[i,j-1]=100000   #on fausse le score du passeur
                else:
                    ScoreTeamates[i,j-1]=pred(df, sender, j,i)
        else:             #equipe 2
            for j in range(15,29):
                if j == sender:
                    ScoreTeamates[i,j-15] = 100000  #on fausse le score du passeur
                else:
                    ScoreTeamates[i,j-15] = pred (df, sender, j,i)
    return ScoreTeamates

#renvoie un vecteur avec toutes les predictions pour une matrice de scores donnée 

def prediction(mat, df):
    NbLignes = df.shape[0]
    prediction = np.zeros(NbLignes)
    for i in range(NbLignes):
        if df.iloc[i,2]<14:
            prediction[i] = np.argmin(mat[i,:])+1
        else:
            prediction[i] = np.argmin(mat[i,:])+15
    return prediction

#fonction qui renvoie tous notre dataframe

def creation_dataframe (df):
     #création variable interception

    df["interception"] = np.zeros(len(df["x_1"]))
    for i in range(len(df["x_1"])):
        if ((df["sender_id"][i]< 15) and (df["receiver_id"][i]>14)):    #sender equipe 1
            df["interception"][i] = 1
        if ((df["sender_id"][i]> 14) and (df["receiver_id"][i]<15)):    #sender equipe 2
            df["interception"][i] = 1
    
    #remplacement des nan
    df = df.replace(np.nan, 100000) 
    
    #on enlève les interceptions    
    df = df[df['interception']==0]
    
    #création variable distance de passe
    NbLignes=df.shape[0]
    DistPasse=np.zeros(NbLignes)
    for i in range(NbLignes):
        DistPasse[i]=distance(df,df.iloc[i,2],df.iloc[i,3],i)
    df['DistPasse']=DistPasse
    
    #creation variable temps de passe
    timestart=df["time_start"]
    timeend=df["time_end"]
    timestart=np.array(timestart)
    timeend=np.array(timeend)
    PassTime=-timestart+timeend
    df["PassTime"]=PassTime
    
    #creation variable côté
    SenderTeamSide=["" for x in range(NbLignes)]
    for i in range(NbLignes):

        if df.iloc[i,2]>14:
            SenderTeamSide[i]="Left"
        else:
            SenderTeamSide[i]="Right"
    df['SenderTeamSide']=SenderTeamSide
    

    mat_score1 = matrice_de_prediction (distance, df)
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
    
    
    