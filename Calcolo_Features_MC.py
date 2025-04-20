import pickle
import numpy as np
import scipy as sc
import pandas as pd
from matplotlib import pyplot as plt

with open('dizionario_mc.pkl', 'rb') as file:
    dizionario = pickle.load(file)

# DETERMINO LE FEATURES

# Ragiono con la classe

# calcolo le features per paz
dati_nome_file = list(dizionario.keys())
dati_skeleton = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste

# prendo i dati della vf 2

indici_vf2 = []
for indice, nomi in enumerate(dati_nome_file):
    if "C002" in nomi:
        if "R002" in nomi:
            indici_vf2.append(indice)

dati_nome_file_vf2 = []
dati_skeleton_vf2 = []
for i in indici_vf2:
    dati_nome = dati_nome_file[i]
    dati_skel = dati_skeleton[i]

    dati_nome_file_vf2.append(dati_nome)
    dati_skeleton_vf2.append(dati_skel)


class statisticheSkeleton_finestra:
    def __init__(self,data):
        self.data = data

    def calcolo_media(self):
        #lista = []
        #for i in range(75):
            #colonna = self.data[:, i]
            #dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            #media1 = np.mean(colonna[0:dim[0]])
            #media2 = np.mean(colonna[dim[0]:dim[1]])
            #media3 = np.mean(colonna[dim[1]:dim[2]])
            #media4 = np.mean(colonna[dim[2]:dim[3]])
            #lista.append((media1, media2, media3, media4))

        #flat_array = np.array(lista).transpose(1, 0)

        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            media_valore = np.mean(colonna)
            lista.append(media_valore)

        return lista

    def calcolo_deviazione_std(self):
        #lista = []
        #for i in range(75):
        #    colonna = self.data[:, i]
        #    dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
        #    std1 = np.std(colonna[0:dim[0]])
        #    std2 = np.std(colonna[dim[0]:dim[1]])
        #    std3 = np.std(colonna[dim[1]:dim[2]])
        #    std4 = np.std(colonna[dim[2]:dim[3]])
        #    lista.append((std1,std2,std3,std4))

        #flat_array = np.array(lista).transpose(1,0)

        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            deviazione_std = np.std(colonna)
            lista.append(deviazione_std)


        return lista


    def calcolo_massimo(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            max1 = np.max(colonna[0:dim[0]])
            max2 = np.max(colonna[dim[0]:dim[1]])
            max3 = np.max(colonna[dim[1]:dim[2]])
            max4 = np.max(colonna[dim[2]:dim[3]])
            lista.append((max1, max2, max3, max4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array


    def calcolo_minimo(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            min1 = np.min(colonna[0:dim[0]])
            min2 = np.min(colonna[dim[0]:dim[1]])
            min3 = np.min(colonna[dim[1]:dim[2]])
            min4 = np.min(colonna[dim[2]:dim[3]])
            lista.append((min1, min2, min3, min4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array


    def calcolo_range(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            range1 = np.max(colonna[0:dim[0]]) - np.min(colonna[0:dim[0]])
            range2 = np.max(colonna[dim[0]:dim[1]]) - np.min(colonna[dim[0]:dim[1]])
            range3 = np.max(colonna[dim[1]:dim[2]]) - np.min(colonna[dim[1]:dim[2]])
            range4 = np.max(colonna[dim[2]:dim[3]]) - np.min(colonna[dim[2]:dim[3]])
            lista.append((range1, range2, range3, range4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_mediana(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            med1 = np.median(colonna[0:dim[0]])
            med2 = np.median(colonna[dim[0]:dim[1]])
            med3 = np.median(colonna[dim[1]:dim[2]])
            med4 = np.median(colonna[dim[2]:dim[3]])
            lista.append((med1, med2, med3, med4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_median_absolute_deviation(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            med1 = np.median(np.abs(colonna[0:dim[0]] - np.median(colonna)))
            med2 = np.median(np.abs(colonna[dim[0]:dim[1]] - np.median(colonna)))
            med3 = np.median(np.abs(colonna[dim[1]:dim[2]] - np.median(colonna)))
            med4 = np.median(np.abs(colonna[dim[2]:dim[3]] - np.median(colonna)))
            lista.append((med1, med2, med3, med4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_absolute_harmonic_mean(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]

            for j in range(len(colonna)):
                if colonna[j] == 0:
                    colonna[j] = 1e-10

            val_AHM1 = len(colonna[0:dim[0]]) / sum(1 / np.abs(colonna[0:dim[0]]))
            val_AHM2 = len(colonna[dim[0]:dim[1]]) / sum(1 / np.abs(colonna[dim[0]:dim[1]]))
            val_AHM3 = len(colonna[dim[1]:dim[2]]) / sum(1 / np.abs(colonna[dim[1]:dim[2]]))
            val_AHM4 = len(colonna[dim[2]:dim[3]]) / sum(1 / np.abs(colonna[dim[2]:dim[3]]))


            lista.append((val_AHM1, val_AHM2, val_AHM3, val_AHM4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_range_IQ(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            range_IQ1 = np.percentile(np.sort(colonna[0:dim[0]]), 75) - np.percentile(np.sort(colonna[0:dim[0]]), 25)
            range_IQ2 = np.percentile(np.sort(colonna[dim[0]:dim[1]]),75) - np.percentile(np.sort(colonna[dim[0]:dim[1]]),25)
            range_IQ3 = np.percentile(np.sort(colonna[dim[1]:dim[2]]),75) - np.percentile(np.sort(colonna[dim[1]:dim[2]]),25)
            range_IQ4 = np.percentile(np.sort(colonna[dim[2]:dim[3]]),75) - np.percentile(np.sort(colonna[dim[2]:dim[3]]),25)
            lista.append((range_IQ1, range_IQ2, range_IQ3, range_IQ4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_mean_velocity(self):
        #lista = []
        #for i in range(75):
        #    colonna = self.data[:, i]
        #    dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
        #    mean_velocity1 = np.mean(np.diff(colonna[0:dim[0]]))
        #    mean_velocity2 = np.mean(np.diff(colonna[dim[0]:dim[1]]))
        #    mean_velocity3 = np.mean(np.diff(colonna[dim[1]:dim[2]]))
        #    mean_velocity4 = np.mean(np.diff(colonna[dim[2]:dim[3]]))
        #    lista.append((mean_velocity1, mean_velocity2, mean_velocity3, mean_velocity4))

        #flat_array = np.array(lista).transpose(1, 0)

        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            mean_velocity_val = np.mean(np.diff(colonna))
            lista.append(mean_velocity_val)

        return lista

    def calcolo_mean_energy(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            mean_energy1 = (1/len(colonna[0:dim[0]]))*sum(np.square(colonna[0:dim[0]])) # np.power(colonna,2)
            mean_energy2 = (1 / len(colonna[dim[0]:dim[1]])) * sum(np.square(colonna[dim[0]:dim[1]]))
            mean_energy3 = (1 / len(colonna[dim[1]:dim[2]])) * sum(np.square(colonna[dim[1]:dim[2]]))  # np.power(colonna,2)
            mean_energy4 = (1 / len(colonna[dim[2]:dim[3]])) * sum(np.square(colonna[dim[2]:dim[3]]))
            lista.append((mean_energy1, mean_energy2, mean_energy3, mean_energy4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_kurtosis(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            noise1 = np.random.normal(0, 1e-10, size=len(colonna[0:dim[0]]))
            kurtosi1 =  sc.stats.kurtosis(colonna[0:dim[0]] + noise1)
            noise2 = np.random.normal(0, 1e-10, size=len(colonna[dim[0]:dim[1]]))
            kurtosi2 = sc.stats.kurtosis(colonna[dim[0]:dim[1]] + noise2)
            noise3 = np.random.normal(0, 1e-10, size=len(colonna[dim[1]:dim[2]]))
            kurtosi3 = sc.stats.kurtosis(colonna[dim[1]:dim[2]] + noise3)
            noise4 = np.random.normal(0, 1e-10, size=len(colonna[dim[2]:dim[3]]))
            kurtosi4 = sc.stats.kurtosis(colonna[dim[2]:dim[3]] + noise4)
            lista.append((kurtosi1, kurtosi2, kurtosi3, kurtosi4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_skewness(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            noise1 = np.random.normal(0, 1e-10, size=len(colonna[0:dim[0]]))
            skewness1 = sc.stats.skew(colonna[0:dim[0]]+noise1)
            noise2 = np.random.normal(0, 1e-10, size=len(colonna[dim[0]:dim[1]]))
            skewness2 = sc.stats.skew(colonna[dim[0]:dim[1]]+noise2)
            noise3 = np.random.normal(0, 1e-10, size=len(colonna[dim[1]:dim[2]]))
            skewness3 = sc.stats.skew(colonna[dim[1]:dim[2]]+noise3)
            noise4 = np.random.normal(0, 1e-10, size=len(colonna[dim[2]:dim[3]]))
            skewness4 = sc.stats.skew(colonna[dim[2]:dim[3]]+noise4)
            lista.append((skewness1, skewness2, skewness3, skewness4))

        flat_array = np.array(lista).transpose(1, 0)

        return flat_array

    def calcolo_coeff_pearson(self):
        lista = []
        index_iniziale = 0
        index_finale = 3
        i = 0

        for i in range(25):
            colonna = self.data[:, index_iniziale:index_finale]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            colonna1 = colonna[0:dim[0], 0:3]
            colonna2 = colonna[dim[0]:dim[1], 0:3]
            colonna3 = colonna[dim[1]:dim[2], 0:3]
            colonna4 = colonna[dim[2]:dim[3], 0:3]

            coeff_pearson1 = np.corrcoef(colonna1, rowvar=False)  # con rowvar false significa la colonna è la variabile e le righe sono le osservazioni
            coeff_pearson1 = coeff_pearson1[0,::]
            coeff_pearson2 = np.corrcoef(colonna2, rowvar=False)
            coeff_pearson2 = coeff_pearson2[0, ::]
            coeff_pearson3 = np.corrcoef(colonna3, rowvar=False)
            coeff_pearson3 = coeff_pearson3[0, ::]
            coeff_pearson4 = np.corrcoef(colonna4, rowvar=False)
            coeff_pearson4 = coeff_pearson4[0, ::]

            index_finale += 3
            index_iniziale += 3

            lista.append((coeff_pearson1, coeff_pearson2, coeff_pearson3, coeff_pearson4))

        lista_array = np.array(lista)
        flatten_lista = lista_array.transpose(1, 0, 2).reshape(4, 75)

        return flatten_lista


liste_mean = []
liste_std = []
#liste_max = []
#liste_min = []
#liste_range = []
#liste_mediana = []
#liste_MAD = []
#liste_AHM = []
#liste_rangeIQ = []
liste_mean_velocity = []
#liste_mean_energy = []
#liste_kurtosi = []
#liste_skew = []
#liste_coeffpears = []

for data in dati_skeleton_vf2:
    valori_fin = statisticheSkeleton_finestra(data)
    liste_std.append(valori_fin.calcolo_deviazione_std())
    liste_mean.append(valori_fin.calcolo_media())
    #liste_max.append(valori_fin.calcolo_massimo())
    #liste_min.append(valori_fin.calcolo_minimo())
    #liste_range.append(valori_fin.calcolo_range())
    #liste_mediana.append(valori_fin.calcolo_mediana())
    #liste_MAD.append(valori_fin.calcolo_median_absolute_deviation())
    #liste_AHM.append(valori_fin.calcolo_absolute_harmonic_mean())
    #liste_rangeIQ.append(valori_fin.calcolo_range_IQ())
    liste_mean_velocity.append(valori_fin.calcolo_mean_velocity())
    #liste_mean_energy.append(valori_fin.calcolo_mean_energy())
    #liste_kurtosi.append(valori_fin.calcolo_kurtosis())
    #liste_skew.append(valori_fin.calcolo_skewness())
    #liste_coeffpears.append(valori_fin.calcolo_coeff_pearson())

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
#matrice_pears = np.array(liste_coeffpears)

# dopo il calcolo delle features dovrò procedere alla suddivisione di training e test

# parto col concatenare le matrici

#matrix_concatenata = np.concatenate((matrice_medie, matrice_std, matrice_mean_velocity, matrice_mediana, matrice_max, matrice_min, matrice_range,
#                                    matrice_AHM,matrice_AHM, matrice_rangeIQ,matrice_mean_energy,matrice_kurtosi, matrice_skew,matrice_pears), axis = 2)


#reshape_matrix = matrix_concatenata.reshape(matrix_concatenata.shape[0],-1)
#np.save('matrice_features_mc.npy', matrix_concatenata)

#with open('nome_file_vf2.pkl','wb') as f:
#    pickle.dump(dati_nome_file_vf2,f)


for index, nome in enumerate(dati_nome_file_vf2):

    if "C002" in nome:
        if "R002" in nome:
            if "P001" in nome:
                if "S001" in nome:
                    if "A008" in nome:
                        indice_classe8_paz1 = index
                        print(indice_classe8_paz1)


for index, nome in enumerate(dati_nome_file_vf2):

    if "C002" in nome:
        if "R002" in nome:
            if "P001" in nome:
                if "S001" in nome:
                    if "A009" in nome:
                        indice_classe9_paz1 = index
                        print(indice_classe9_paz1)



for index, nome in enumerate(dati_nome_file_vf2):

    if "C002" in nome:
        if "R002" in nome:
            if "P001" in nome:
                if "S001" in nome:
                    if "A046" in nome:
                        indice_classe46_paz1 = index
                        print(indice_classe46_paz1)


for index, nome in enumerate(dati_nome_file_vf2):

    if "C002" in nome:
        if "R002" in nome:
            if "P001" in nome:
                if "S001" in nome:
                    if "A060" in nome:
                        indice_classe60_paz1 = index
                        print(indice_classe60_paz1)



media_paz1_cl8 = matrice_medie[indice_classe8_paz1]
media_paz1_cl9 = matrice_medie[indice_classe9_paz1]

std_paz1_cl8 = matrice_std[indice_classe8_paz1]
std_paz1_cl9 = matrice_std[indice_classe9_paz1]

mean_vel_paz1_cl8 = matrice_mean_velocity[indice_classe8_paz1]
mean_vel_paz1_cl9 = matrice_mean_velocity[indice_classe9_paz1]



media_paz1_cl46 = matrice_medie[indice_classe46_paz1]
media_paz1_cl60 = matrice_medie[indice_classe60_paz1]

std_paz1_cl46 = matrice_std[indice_classe46_paz1]
std_paz1_cl60 = matrice_std[indice_classe60_paz1]

mean_vel_paz1_cl46 = matrice_mean_velocity[indice_classe46_paz1]
mean_vel_paz1_cl60 = matrice_mean_velocity[indice_classe60_paz1]



figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols=1, sharex = True)
ax1.bar(np.arange(75), media_paz1_cl8)
ax2.bar(np.arange(75), media_paz1_cl9)
ax3.bar(np.arange(75), media_paz1_cl46)
ax4.bar(np.arange(75), media_paz1_cl60)

plt.show()

figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols=1, sharex = True)
ax1.bar(np.arange(75), std_paz1_cl8)
ax2.bar(np.arange(75), std_paz1_cl9)
ax3.bar(np.arange(75), std_paz1_cl46)
ax4.bar(np.arange(75), std_paz1_cl60)

plt.show()


figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols=1, sharex = True)
ax1.bar(np.arange(75), mean_vel_paz1_cl8)
ax2.bar(np.arange(75), mean_vel_paz1_cl9)
ax3.bar(np.arange(75), mean_vel_paz1_cl46)
ax4.bar(np.arange(75), mean_vel_paz1_cl60)
plt.show()





