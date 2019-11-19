import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as func




#Calcul de tous les scores pour deux joueurs 

def Score1(dfligne, sender, receveur) : 
    d = np.sqrt ((dfligne["x_{}".format(int(sender))] - dfligne["x_{}".format(int(receveur))])**2 + (dfligne["y_{}".format(int(sender))] - dfligne["y_{}".format(int(receveur))])**2)
    return d



def Score2 (dfligne, sender, receveur) : 
    score = Score1 (dfligne, sender, receveur)
    if func.perimetre (dfligne, sender, receveur) : 
        score += 900
    return score


def Score3(dfligne, sender, receveur) : 
    score = Score2(dfligne, sender, receveur)
    if func.nombre_adversaires (dfligne, sender, receveur) >= 2:
        score += 55     
    return score

def Score4(dfligne, sender, receveur) : 
    SenderX = dfligne["x_{}".format(sender)]
    ReceiverX = dfligne["x_{}".format(receveur)]
    score = Score3(dfligne, sender, receveur)
    if func.DirectionPasse(dfligne)=="Backward":
        score += 0.1 * np.abs(SenderX - ReceiverX)
    else :
        score -= 0.3 * np.abs(SenderX - ReceiverX)
    return score