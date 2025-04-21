import pickle
import numpy as np
from dtaidistance import dtw
from scipy.special import kl_div
import pandas as pd

with open('dizionario.pkl', 'rb') as f:
    dizionario = pickle.load(f)

skeleton_body = list(dizionario.values())
data_nome = list(dizionario.keys())

indici_vf2 = []
for index, nome in enumerate(data_nome):
    if "C002" in nome:
        if "R002" in nome:
            indici_vf2.append(index)

data_skeleton_body = []
for i in indici_vf2:
    data = skeleton_body[i]
    data_skeleton_body.append(data)

dati_nomi = []
for i in indici_vf2:
    nome = data_nome[i]
    dati_nomi.append(nome)

altezza = [1.7, 1.7, 1.4, 1.2, 1.2, 0.8, 0.5, 1.4, 0.8, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.7, 2.5]
distanza = [3.5, 2.5, 2.5, 3.0, 3.0, 3.5, 4.5, 3.5, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.5, 3.5, 3.0]

feat_altezza = []
feat_distance = []
for nome in dati_nomi:

    if "S001" in nome:

        height = altezza[0]
        distance = distanza[0]

    elif "S002" in nome:

        height = altezza[1]
        distance = distanza[1]

    elif "S003" in nome:

        height = altezza[2]
        distance = distanza[2]

    elif "S004" in nome:

        height = altezza[3]
        distance = distanza[3]

    elif "S005" in nome:

        height = altezza[4]
        distance = distanza[4]

    elif "S006" in nome:

        height = altezza[5]
        distance = distanza[5]

    elif "S007" in nome:

        height = altezza[6]
        distance = distanza[6]

    elif "S008" in nome:

        height = altezza[7]
        distance = distanza[7]

    elif "S009" in nome:

        height = altezza[8]
        distance = distanza[8]

    elif "S010" in nome:

        height = altezza[9]
        distance = distanza[9]

    elif "S011" in nome:

        height = altezza[10]
        distance = distanza[10]

    elif "S012" in nome:

        height = altezza[11]
        distance = distanza[11]

    elif "S013" in nome:

        height = altezza[12]
        distance = distanza[12]

    elif "S014" in nome:

        height = altezza[13]
        distance = distanza[13]

    elif "S015" in nome:

        height = altezza[14]
        distance = distanza[14]

    elif "S016" in nome:

        height = altezza[15]
        distance = distanza[15]

    elif "S017" in nome:

        height = altezza[16]
        distance = distanza[16]

    feat_altezza.append(height)
    feat_distance.append(distance)

feat_altezza = np.array(feat_altezza)
feat_distance = np.array(feat_distance)

feat_altezza = np.transpose(feat_altezza)
feat_distance = np.transpose(feat_distance)

feat_altezza = feat_altezza.reshape(-1,1) # formatto la dimensione
feat_distance = feat_distance.reshape(-1,1)

feat_setting = np.hstack((feat_altezza, feat_distance))

# calcolo la distanza media

def calcolo_distanza_setting(x1):

    dist_setting = []

    i = 0
    for x2 in feat_setting:

        dist = np.abs((x1 - x2) / (x1 + x2 + 1e-10))
        dist_mean = np.mean(dist)
        dist_setting.append(dist_mean)

        i += 1

        print(i)

    return dist_setting


distance_tot = []
for row in feat_setting:

    distanza = calcolo_distanza_setting(row)
    distance_tot.append(distanza)

distance_tot = np.array(distance_tot)
print(distance_tot.shape)

dizionario = {}

for i in range(len(distance_tot)):

    array = distance_tot[i]
    dizionario[dati_nomi[i]] = array

with open('distance_camera.pkl', 'wb') as f:
    pickle.dump(dizionario, f)


