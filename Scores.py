import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as func




#Calcul de tous les scores pour deux joueurs 

def Score1(df, sender, receveur, ligne) : 
    d = np.sqrt ((df["x_{}".format(int(sender))][ligne] - df["x_{}".format(int(receveur))][ligne])**2 + (df["y_{}".format(int(sender))][ligne] - df["y_{}".format(int(receveur))][ligne])**2)
    return d



def Score2 (df, sender, receveur, ligne) : 
    score = Score1 (df, sender, receveur,ligne)
    if func.perimetre (df, sender, receveur, ligne) : 
        score += 900
    return score


def Score3(df, sender, receveur, ligne) : 
    score = Score2(df, sender, receveur,ligne)
    if func.nombre_adversaires (df, sender, receveur, ligne) >= 2:
        score += 55     
    return score

def Score4(df, sender, receveur, ligne) : 
    SenderX = df ["x_{}".format(sender)][ligne]
    ReceiverX = df["x_{}".format(receveur)][ligne]
    score = Score3(df, sender, receveur, ligne)
    
    if func.DirectionPasse(ligne, df)=="Backward":
        score += 0.1 * np.abs(SenderX - ReceiverX)
    else :
        score -= 0.3 * np.abs(SenderX - ReceiverX)
    return score