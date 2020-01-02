import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def image(dfligne,affichage=False):
    image=np.zeros((680,1050,3))
    for i in range(1,15):
        if dfligne["x_{}".format(i)]!=100000:
            x,y=int(dfligne["y_{}".format(i)]/10)+340,int(dfligne["x_{}".format(i)]/10)+525
            image[x-5:x+6,y-5:y+6,0]=np.ones((11,11))*255
    for i in range(15,29):
        if dfligne["x_{}".format(i)]!=100000:
            x,y=int(dfligne["y_{}".format(i)]/10)+340,int(dfligne["x_{}".format(i)]/10)+525
            print(x,y)
            image[x-5:x+6,y-5:y+6,1]=np.ones((11,11))*255
    sender=int(dfligne['sender_id'])
    x,y=int(dfligne["y_{}".format(sender)]/10)+340,int(dfligne["x_{}".format(sender)]/10)+525
    image[x-5:x+6,y-5:y+6,2]=np.ones((11,11))*255
    if affichage == True:
        xmin, xmax, ymin, ymax = -5250, 5250, -3400, 3400
        XBleu=np.zeros(14)
        XRouge=np.zeros(14)
        YBleu=np.zeros(14)
        YRouge=np.zeros(14)
        for i in range(1,15):
            XBleu[i-1],YBleu[i-1]=int(dfligne["x_{}".format(i)]),int(dfligne["y_{}".format(i)])
        for i in range(15,29):
            XRouge[i-15],YRouge[i-15]=int(dfligne["x_{}".format(i)]),int(dfligne["y_{}".format(i)])
        plt.scatter(XBleu, YBleu, color='blue',label="Equipe A")
        plt.scatter(XRouge, YRouge, color='red',label="Equipe B")
        plt.xlim (xmin, xmax)
        plt.ylim (ymin, ymax)
        plt.legend()
        plt.show()
    return(np.flip(image,axis=0))        
        



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
def shift_equipe_partenaire_vecteur (sender) :
    return np.array([i>14 for i in sender])*14

#fonction qui renvoie le décalage nécessaire pour balayer l'équipe du passeur 
def shift_equipe_partenaire (sender) :
    if sender < 15:
        return 0
    else:
        return 14


#On regarde si un adversaire est dans le périmètre ou pas, utile pour score 2, 3 et 4
def perimetre (dfligne, sender, receveur) : 
    Trouve = False
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        if distance(dfligne, receveur, i) < 700 :
            Trouve = True 
            return Trouve
    return Trouve 

#On regarde combien d'adversaires sont dans le périmètre, utile pour score 3 
def nombre_adversaires (dfligne, sender, receveur) : 
    nombre_adversaires = 0
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        if distance(dfligne, receveur, i) < 700:
            nombre_adversaires += 1
    return nombre_adversaires




#Systeme de passe backward/Forward classique : indique si la passe est vers l'avant ou vers l'arrière
def DirectionPasse (dfligne,receiver) :
    sender = dfligne["sender_id"]
    SenderX = dfligne["x_{}".format(int(sender))] 
    ReceiverX = dfligne["x_{}".format(int(receiver))]
    if sender < 15 :  #equipe a droite 
        if SenderX < ReceiverX :
            Direction = 0
        else:
            Direction = 1
    else:  #equipe a gauche
        if SenderX > ReceiverX:
            Direction = 0
        else:
            Direction = 1
    return Direction

#Systeme de passe avec ajout de passes laterales
def DirectionPasse_amelioree (dfligne,receiver) :
    sender = dfligne["sender_id"]
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

def matrice_direction_passe(df):
    NbLignes = df.shape [0]
    matDirPasse=np.zeros((NbLignes,14))
    for j in range (1, 15) :
        matDirPasse[:,j-1]= df.apply(lambda x : DirectionPasse (x,j),axis=1)
    return matDirPasse


#creation de la matrice de score pour chaque coéquipier du passeur, on lui passe en argument si on veur que ce soit le score 1, 2, 3 ou 4

#fonction à appliquer à chaque ligne du dataframe
def matrice_de_prediction_ligne(x,pred,j):
    sender=x['sender_id']
    if sender==j+shift_equipe_partenaire(sender):
        c=100000
    else:
        c=pred(x,sender,j+shift_equipe_partenaire(sender))
    return c

#on applique la fonction d'au-dessus à tout le dataframe
def matrice_de_prediction (pred, df) :
    NbLignes = df.shape [0]
    ScoreTeamates = np.zeros ((NbLignes, 14))
    for j in range (1, 15) :
        ScoreTeamates[:,j-1]= df.apply(lambda x : matrice_de_prediction_ligne (x,pred,j),axis=1)
    return ScoreTeamates


#renvoie un vecteur avec toutes les predictions pour une matrice de scores donnée (calculée avec la fonction d'au-dessus)

def prediction (mat, df) :   #applique a toutes les lignes 
    mat = pd.DataFrame(mat)
    sender = df["sender_id"]
    mat["sender"] = sender
    prediction = mat.apply(lambda x: np.argmin(x[:-1]) + shift_equipe_partenaire(x["sender"]) + 1 , axis = 1)
    return prediction


#création variable interception dans un dataframe
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

#création variable team side dans un dataframe 
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
    df = df.reset_index().drop(["index"], axis=1)
    return df

#CREATION DE FEATURES DE L'ARTICLE 

#fonction qui retourne les distances des 2 plus proches adversaires du sender. on l'ajoute avec un apply dans le dataframe. 
def distances_sender_opponents (dfligne):
    sender = dfligne["sender_id"]
    liste = []
    for i in range (1 + shift_equipe_adverse (sender), 15 + shift_equipe_adverse (sender)) :
        liste += [distance(dfligne, sender, i)]
    result1 = min(liste)
    del liste[np.argmin(liste)]
    result2 = min(liste)
    return result1, result2

#ajout dans le dataframe avec un apply de la fonction au-dessus
def ajout_distances_sender (df):
    df["premiere_distance_sender"] = df.apply(lambda line: distances_sender_opponents(line)[0] , axis = 1)
    df["seconde_distance_sender"] = df.apply(lambda line: distances_sender_opponents(line)[1] , axis = 1)
    return df

#fonction qui retourne les distances des 2 plus proches adversaires d'un receveur potentiel. on l'ajoute ensuite avec un apply dans le dataframe.
def distances_receveur (dfligne, receveur):
    sender = dfligne["sender_id"]
    liste = []
    if (sender == receveur):
        return 0, 0
    else:
        for i in range (1 + shift_equipe_adverse (receveur), 15 + shift_equipe_adverse (receveur)) :
            liste += [distance(dfligne, receveur, i)]
        result1 = min(liste)
        del liste[np.argmin(liste)]
        result2 = min(liste)
        return result1, result2
    
#ajout dans le dataframe avec un apply de la fonction au-dessus
def ajout_distances_receveur (df):
        for i in range(1, 15):
            df["premiere_distances_receveur_{}".format(i)] = df.apply(lambda line: distances_receveur (line, i + shift_equipe_partenaire(line["sender_id"]))[0], axis = 1)
            df["seconde_distances_receveur_{}".format(i)] = df.apply(lambda line: distances_receveur (line, i + shift_equipe_partenaire(line["sender_id"]))[1], axis = 1)
        return df 

#on calcule la distance minimale entre un adversaire et la ligne de passe pour un receveur potentiel, on l'ajoute ensuite dans le dataframe avec un apply
def distance_ligne_passe (dfligne,receveur):
    sender = dfligne['sender_id']
    Xsender = dfligne["x_{}".format(int(sender))]
    Ysender = dfligne["y_{}".format(int(sender))]
    Xreceveur = dfligne["x_{}".format(int(receveur))]
    Yreceveur = dfligne["y_{}".format(int(receveur))]
    liste = []
    if (sender == receveur):
        return 0
    elif (Xreceveur == 100000):
        return 0
    else:
        for adversaire in range(1 + shift_equipe_adverse (receveur), 15 + shift_equipe_adverse (receveur)):
            Xadversaire = dfligne["x_{}".format(int(adversaire))]
            Yadversaire = dfligne["y_{}".format(int(adversaire))]
            a = np.array([Xreceveur,Yreceveur])
            b = np.array([Xsender,Ysender])
            c = np.array([Xadversaire,Yadversaire])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            liste += [np.sin(angle) * distance(dfligne, sender, adversaire) if angle*180/np.pi < 90 and angle*180/np.pi > (-90) else 100000]
        result = min(liste)
    
        return(result)

#ajout dans le dataframe avec un apply de la fonction au-dessus
def ajout_distance_ligne_passe (df):
    for i in range (1, 15):
        df["distance_ligne_passe_{}".format(i)] = df.apply(lambda line: distance_ligne_passe (line, i + shift_equipe_partenaire(line["sender_id"])), axis = 1)
    return df
        
        

#On regarde si un adversaire est présent dans un cône d'un angle minimum par rapport à la ligne de passe
def adversaire_dans_cone (dfligne, receveur, demiangle, affichage = False):
    result = False
    sender = dfligne['sender_id']
    receveur += shift_equipe_partenaire(sender)
    Xsender = dfligne["x_{}".format(int(sender))]
    Ysender = dfligne["y_{}".format(int(sender))]
    Xreceveur = dfligne["x_{}".format(int(receveur))]
    Yreceveur = dfligne["y_{}".format(int(receveur))]
    a = np.array([Xreceveur, Yreceveur])
    b = np.array([Xsender, Ysender])
    for i in range(1 + shift_equipe_adverse(sender), 15 + shift_equipe_adverse (sender)):
        Xadversaire = dfligne["x_{}".format(int(i))]
        Yadversaire = dfligne["y_{}".format(int(i))]

        c = np.array([Xadversaire,Yadversaire])
        
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)*180/np.pi
        if affichage == True:
            xmin, xmax, ymin, ymax = -5250, 5250, -3400, 3400
            plt.scatter(Xsender, Ysender, color='blue',label="sender")
            plt.scatter(Xreceveur, Yreceveur, color='green',label="receveur")
            plt.scatter(Xadversaire, Yadversaire, color='red',label="adversaire")
            plt.xlim (xmin, xmax)
            plt.ylim (ymin, ymax)
            plt.legend()
            plt.show()
        if (abs(angle) < demiangle and np.linalg.norm(c-a)<np.linalg.norm(ba)) or np.linalg.norm(c-a)<300 :
            result = True
            break
    if sender == receveur:
        result = False
    return result

#construction de la matrice qui indique pour chaque receveur potentiel si un adversaire est présent dans le cône de l'angle voulu (utilise la fonction d'au dessus)
def Matrice_adversaire_dans_cone(df,demiangle):
    Nblignes = df.shape [0]
    MatriceIntercept=np.zeros((Nblignes,14))
    MatriceIntercept=np.array(MatriceIntercept,dtype=bool)
    for i in range(1,15):
        MatriceIntercept[:,i-1]=df.apply(lambda x  : adversaire_dans_cone(x,i,demiangle,False),axis=1)
    return MatriceIntercept
    