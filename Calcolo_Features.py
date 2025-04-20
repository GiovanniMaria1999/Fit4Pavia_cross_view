import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pd
from matplotlib import pyplot

with open('dizionario.pkl', 'rb') as file:
    dizionario = pickle.load(file)

# estrazione paz

def estrazione_nome_chiave(s):
    inizio = s.index("P")
    fine = inizio+4
    return s[inizio:fine]

nome_chiave = []
for chiave in dizionario:
    substring = estrazione_nome_chiave(chiave)
    if substring in chiave:
        nome_chiave.append(chiave)

# DETERMINO LE FEATURES

# Ragiono con la classe

# calcolo le features per paz
dati_nome_file = list(dizionario.keys())
dati_skeleton = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste

class Statistiche_Skel:
    def __init__(self,data):
        self.data = data

    def media(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            media_valore = np.mean(colonna)
            lista.append(media_valore)
        return lista

    def deviazione_std(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            deviazione_std = np.std(colonna)
            lista.append(deviazione_std)
        return lista

    def val_massimo(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            massimo = np.max(colonna)
            lista.append(massimo)
        return lista

    def val_minimo(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            minimo = np.min(colonna)
            lista.append(minimo)
        return lista

    def range(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            range_val = np.max(colonna) - np.min(colonna)
            lista.append(range_val)
        return lista

    def mediana(self):
        lista = []
        for i in range(75):
            colonna = np.sort(self.data[:, i])
            val_median = np.median(colonna)
            lista.append(val_median)
        return lista

    def median_absolute_deviation(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            val_MAD = np.median(np.sort(np.abs(colonna - np.median(np.sort(colonna)))))
            lista.append(val_MAD)
        return lista

    def absolute_harmonic_mean(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            for j in range(len(colonna)):
                if colonna[j] == 0:
                    colonna[j] = 1e-10

            val_AHM = len(colonna)/sum(1/np.abs(colonna))
            lista.append(val_AHM)
        return lista

    def range_IQ(self):
        lista = []
        for i in range(75):
            colonna = np.sort(self.data[:, i])
            range_IQ_val = np.percentile(colonna, 75) - np.percentile(colonna, 25)
            lista.append(range_IQ_val)
        return lista

    def mean_velocity(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            mean_velocity_val = np.mean(np.diff(colonna))
            lista.append(mean_velocity_val)
        return lista

    def mean_energy(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            mean_energy_val = (1/len(colonna))*sum(np.square(colonna)) # np.power(colonna,2)
            lista.append(mean_energy_val)
        return lista

    def kurtosis(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            kurtosi_val =  sc.stats.kurtosis(colonna)
            lista.append(kurtosi_val)
        return lista

    def skewness(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            skewness_val = sc.stats.skew(colonna)
            lista.append(skewness_val)

        return lista

    def pearson(self):
        lista = []
        index_finale =  3
        index_iniziale = 0
        while index_finale < 76:
            colonna = self.data[:,index_iniziale:index_finale]
            coeff_pearson = np.corrcoef(colonna, rowvar = False) # con rowvar false significa la colonna è la variabile e le righe sono le osservazioni
            coeff_pearson = coeff_pearson[0,::]

            index_finale += 3
            index_iniziale += 3
            lista.append(coeff_pearson)

        flat_data = np.array(lista).flatten()[:75]

        return flat_data

liste_mean = []
liste_std = []
liste_max = []
liste_min = []
liste_range = []
liste_mediana = []
liste_MAD = []
liste_AHM = []
liste_rangeIQ = []
liste_mean_velocity = []
liste_mean_energy = []
liste_kurtosi = []
liste_skew = []
liste_coeff_pears = []

for data in dati_skeleton:
    valori = Statistiche_Skel(data)
    liste_std.append(valori.deviazione_std())
    liste_mean.append(valori.media())
    #liste_max.append(valori.val_massimo())
    #liste_min.append(valori.val_minimo())
    #liste_range.append(valori.range())
    #liste_mediana.append(valori.mediana())
    #liste_MAD.append(valori.median_absolute_deviation())
    #liste_AHM.append(valori.absolute_harmonic_mean())
    #liste_rangeIQ.append(valori.range_IQ())
    liste_mean_velocity.append(valori.mean_velocity())
    #liste_mean_energy.append(valori.mean_energy())
    #liste_kurtosi.append(valori.kurtosis())
    #liste_skew.append(valori.skewness())
    #liste_coeff_pears.append(valori.pearson())



matrice_medie = np.array(liste_mean)
matrice_std = np.array(liste_std)
#matrice_max = np.array(liste_max)
#matrice_min = np.array(liste_min)
#matrice_range = np.array(liste_range)
#matrice_mediana = np.array(liste_mediana)
#matrice_MAD = np.array(liste_MAD)
#matrice_AHM = np.array(liste_AHM)
#matrice_rangeIQ = np.array(liste_rangeIQ)
matrice_mean_velocity = np.array(liste_mean_velocity)
#matrice_mean_energy = np.array(liste_mean_energy)
#matrice_kurtosi = np.array(liste_kurtosi)
#matrice_skew = np.array(liste_skew)
#matrice_pears = np.array(liste_coeff_pears)



# dopo il calcolo delle features dovrò procedere alla suddivisione di training e test

# parto col concatenare le matrici

#matrix_concatenata = np.concatenate((matrice_medie, matrice_std, matrice_mean_velocity, matrice_mediana, matrice_max, matrice_min, matrice_range,
#                                    matrice_AHM,matrice_AHM, matrice_rangeIQ,matrice_mean_energy,matrice_kurtosi, matrice_skew,matrice_pears), axis = 1)

#tabella_features = pd.DataFrame(matrix_concatenata,index = nome_chiave)
#tabella_features.to_csv("tabella_features.csv")


for index, nome in enumerate(dati_nome_file):

    if "C002" in nome:
        if "R002" in nome:
            if "P001" in nome:
                if "S001" in nome:
                    if "A008" in nome:
                        indice_classe8_paz1 = index


for index, nome in enumerate(dati_nome_file):

    if "C002" in nome:
        if "R002" in nome:
            if "P001" in nome:
                if "S001" in nome:
                    if "A009" in nome:
                        indice_classe9_paz1 = index



media_paz1_cl8 = matrice_medie[indice_classe8_paz1]
media_paz1_cl9 = matrice_medie[indice_classe9_paz1]

std_paz1_cl8 = matrice_std[indice_classe8_paz1]
std_paz1_cl9 = matrice_std[indice_classe9_paz1]

mean_vel_paz1_cl8 = matrice_mean_velocity[indice_classe8_paz1]
mean_vel_paz1_cl9 = matrice_mean_velocity[indice_classe9_paz1]


figure, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
ax1.plot(np.arange(75), media_paz1_cl8)
ax2.plot(np.arange(75), media_paz1_cl9)
plt.show()

figure, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
ax1.plot(np.arange(75), std_paz1_cl8)
ax2.plot(np.arange(75), std_paz1_cl9)
plt.show()


figure, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
ax1.plot(np.arange(75), mean_vel_paz1_cl8)
ax2.plot(np.arange(75), mean_vel_paz1_cl9)
plt.show()





figure, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
ax1.bar(np.arange(75), media_paz1_cl8)
ax2.bar(np.arange(75), media_paz1_cl9)
plt.show()

figure, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
ax1.bar(np.arange(75), std_paz1_cl8)
ax2.bar(np.arange(75), std_paz1_cl9)
plt.show()


figure, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
ax1.bar(np.arange(75), mean_vel_paz1_cl8)
ax2.bar(np.arange(75), mean_vel_paz1_cl9)
plt.show()










