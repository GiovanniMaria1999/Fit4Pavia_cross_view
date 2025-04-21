import pickle
import numpy as np
from dtaidistance import dtw
from scipy.special import kl_div
import pandas as pd


with open('dizionario.pkl', 'rb') as file:
    dizionario = pickle.load(file)



# estrazione paz


nome_chiave = list(dizionario.keys())
dati_skeleton = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste

nome_vista_fron2 = []
indici_vista_fron2 = []
for index, nome in enumerate(nome_chiave):
    if "C002" in nome:
        if "R002" in nome:
            indici_vista_fron2.append(index)


dati_skeleton_body = []
for i in indici_vista_fron2:
    dati = dati_skeleton[i]
    dati_skeleton_body.append(dati)


dati_nome_chiave = []
for i in indici_vista_fron2:
    dati = nome_chiave[i]
    dati_nome_chiave.append(dati)


# calcolo le channel-level-difference

def calcolo_distanza_channel_level(data_paz,type_distanza):

    if type_distanza == "dtw_distance":
        dist_channel = []
        dist_paz = []

        for seq_paz in dati_skeleton_body:
            for j in range(75):
                dist = dtw.distance(data_paz[:,j], seq_paz[:,j])
                dist_channel.append(dist)

            mean_channel = np.mean(dist_channel)
            dist_paz.append(mean_channel)
            dist_channel = []

        return dist_paz

    elif type_distanza == "dtw_distance_klb":
        dist_channel = []
        dist_paz = []


        for seq_paz in dati_skeleton_body:
            for j in range(75):
                dist = dtw.lb_keogh(data_paz[:, j], seq_paz[:, j])
                dist_channel.append(dist)

            mean_channel = np.mean(dist_channel)
            dist_paz.append(mean_channel)
            dist_channel = []


        return dist_paz

    elif type_distanza == "signals_normalised_cross_correlation":
        dist_channel = []
        dist_paz = []

        for seq_paz in dati_skeleton_body:




            for j in range(75):

                dist = np.correlate(data_paz[:, j], seq_paz[:, j], mode='full')/(np.std(data_paz[:, j])*np.std(seq_paz[:, j])*len(data_paz[:,j]))
                dist_channel.append(dist)

            mean_channel = np.mean(dist_channel)
            dist_paz.append(mean_channel)
            dist_channel = []

        return dist_paz

    elif type_distanza == "kl_divergence":

        dist_channel = []
        dist_paz = []
        i = 0
        for seq_paz in dati_skeleton_body:
            i += 1
            for j in range(75):
                hist1, bin_edges1 = np.histogram(data_paz[:, j], bins=100, density=True)
                p = hist1 + 1e-10
                hist2, bin_edges2 = np.histogram(seq_paz[:, j], bins=100, density=True)
                q = hist2 + 1e-10

                dist = kl_div(p, q)
                dist_channel.append(dist)

            mean_channel = np.mean(dist_channel)
            dist_paz.append(mean_channel)
            dist_channel = []
            print(i)

        return dist_paz


distance_tot = []
for data in dati_skeleton_body:
    distance = calcolo_distanza_channel_level(data,"signals_normalised_cross_correlation")
    distance_tot.append(distance)

#array_distance = np.array(distance_tot)
#print(array_distance.shape)

#dizionario_dist = {}

#for i in range(len(array_distance)):

#    array = array_distance[i]
#    dizionario_dist[dati_nome_chiave[i]] = array

#print(len(dizionario_dist))

#with open('distance_kl_divergence.pkl', 'wb') as f:
#    pickle.dump(dizionario_dist, f)


# adesso calcolo la distanza tra feature

# estraggo solo le feature della cam


Tabella_features = pd.read_csv("tabella_features.csv")

indici_vf2 = []
for index, row in Tabella_features.iterrows():
    if "C002" in row.iloc[0]:
        if "R002" in row.iloc[0]:
            indici_vf2.append(index)

tabella_vf2 = Tabella_features.iloc[indici_vf2,:]
tabella_vf2.reset_index(drop=True, inplace=True)
tabella_vf2.index = tabella_vf2.index + 1

tabella_vf2 = tabella_vf2.iloc[:,1::]

def calcolo_distanza_feature(x1):

    dist_paz = []
    i = 0
    for index, x2 in tabella_vf2.iterrows():


        dist = np.abs((x1 - x2)/(x1 + x2 + 1e-10))
        dist_mean = np.mean(dist)
        dist_paz.append(dist_mean)

        i += 1
        print(i)

    return dist_paz

distance_tot = []
for index, row in tabella_vf2.iterrows():
    distanza = calcolo_distanza_feature(row)
    distance_tot.append(distanza)

array_distance_features = np.array(distance_tot)
print(array_distance_features.shape)

dizionario_dist = {}

for i in range(len(array_distance_features)):

    array = array_distance_features[i]
    dizionario_dist[dati_nome_chiave[i]] = array

print(len(dizionario_dist))

with open('distance_features.pkl', 'wb') as f:
    pickle.dump(dizionario_dist, f)





