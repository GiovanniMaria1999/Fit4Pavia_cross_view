import pickle
import numpy as np
import scipy as sc

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

print(nome_chiave)
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
            minimo = np.max(colonna)
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
            val_MAD = np.median(np.abs(colonna - np.median(np.sort(colonna))))
            lista.append(val_MAD)
        return lista

    def absolute_harmonic_mean(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            if np.all(self.data[:, i] != 0): # se tutti gli elementi dell'array sono True
                val_AHM = len(colonna)/sum(1/np.abs(colonna))
            else:
                val_AHM = float('inf')
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

for data in dati_skeleton:
    valori = Statistiche_Skel(data)
    #liste_std.append(valori.deviazione_std())
    liste_mean.append(valori.media())
    #liste_max.append(valori.val_massimo())
    #liste_min.append(valori.val_minimo())
    #liste_range.append(valori.range())
    #liste_mediana.append(valori.mediana())
    #liste_MAD.append(valori.median_absolute_deviation())
    #liste_AHM.append(valori.absolute_harmonic_mean())
    #liste_rangeIQ.append(valori.range_IQ())
    #liste_mean_velocity.append(valori.mean_velocity())
    #liste_mean_energy.append(valori.mean_energy())
    #liste_kurtosi.append(valori.kurtosis())
    #liste_skew.append(valori.skewness())

matrice_medie = np.array(liste_mean)
#np.savetxt('matrice_media.txt', matrice_medie) # questa per matrice 1d/2d
#matrice_std = np.array(liste_std)
#matrice_max = np.array(liste_max)
#matrice_min = np.array(liste_min)
#matrice_range = np.array(liste_range)
#matrice_mediana = np.array(liste_mediana)
#matrice_MAD = np.array(liste_MAD)
#matrice_AHM = np.array(liste_AHM)
#matrice_rangeIQ = np.array(liste_rangeIQ)
#matrice_mean_velocity = np.array(liste_mean_velocity)
#matrice_mean_energy = np.array(liste_mean_energy)
#matrice_kurtosi = np.array(liste_kurtosi)
#matrice_skew = np.array(liste_skew)


# calcolare la feature per paziente impostando delle finestre temporali
# potrei decidere di lavorare con 4 finestre temporali

#for data in dati_skeleton:
    #print(data.shape)


class statisticheSkeleton_finestra:
    def __init__(self,data):
        self.data = data

    def calcolo_media(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            media1 = np.mean(colonna[0:dim[0]])
            media2 = np.mean(colonna[dim[0]:dim[1]])
            media3 = np.mean(colonna[dim[1]:dim[2]])
            media4 = np.mean(colonna[dim[2]:dim[3]])
            lista.append((media1, media2, media3, media4))
        return lista

    def calcolo_deviazione_std(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            std1 = np.std(colonna[0:dim[0]])
            std2 = np.std(colonna[dim[0]:dim[1]])
            std3 = np.std(colonna[dim[1]:dim[2]])
            std4 = np.std(colonna[dim[2]:dim[3]])
            lista.append((std1,std2,std3,std4))
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
        return lista

    def calcolo_minimo(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            min1 = np.std(colonna[0:dim[0]])
            min2 = np.std(colonna[dim[0]:dim[1]])
            min3 = np.std(colonna[dim[1]:dim[2]])
            min4 = np.std(colonna[dim[2]:dim[3]])
            lista.append((min1, min2, min3, min4))
        return lista

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
        return lista

    def calcolo_mediana(self):
        lista = []
        for i in range(75):
            colonna = np.sort(self.data[:, i])
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            med1 = np.median(colonna[0:dim[0]])
            med2 = np.median(colonna[dim[0]:dim[1]])
            med3 = np.median(colonna[dim[1]:dim[2]])
            med4 = np.median(colonna[dim[2]:dim[3]])
            lista.append((med1, med2, med3, med4))
        return lista

    def calcolo_median_absolute_deviation(self):
        lista = []
        for i in range(75):
            colonna = np.sort(self.data[:, i])
            mediana = np.median(colonna)
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            med1 = np.median(np.abs(colonna[0:dim[0]] - mediana))
            med2 = np.median(np.abs(colonna[dim[0]:dim[1]] - mediana))
            med3 = np.median(np.abs(colonna[dim[1]:dim[2]] - mediana))
            med4 = np.median(np.abs(colonna[dim[2]:dim[3]] - mediana))
            lista.append((med1, med2, med3, med4))
        return lista

    def calcolo_absolute_harmonic_mean(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            if np.all(colonna != 0):
                val_AHM1 = len(colonna[0:dim[0]]) / sum(1 / np.abs(colonna[0:dim[0]]))
                val_AHM2 = len(colonna[dim[0]:dim[1]]) / sum(1 / np.abs(colonna[dim[0]:dim[1]]))
                val_AHM3 = len(colonna[dim[1]:dim[2]]) / sum(1 / np.abs(colonna[dim[1]:dim[2]]))
                val_AHM4 = len(colonna[dim[2]:dim[3]]) / sum(1 / np.abs(colonna[dim[2]:dim[3]]))
            else:
                val_AHM = float('inf')
            lista.append(val_AHM1, val_AHM2, val_AHM3, val_AHM4)
        return lista

    def calcolo_range_IQ(self):
        lista = []
        for i in range(75):
            colonna = np.sort(self.data[:, i])
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            range_IQ1 = np.percentile(colonna[0:dim[0]], 75) - np.percentile(colonna[0:dim[0]], 25)
            range_IQ2 = np.percentile(colonna[dim[0]:dim[1]],75) - np.percentile(colonna[dim[0]:dim[1]],25)
            range_IQ3 = np.percentile(colonna[dim[1]:dim[2]],75) - np.percentile(colonna[dim[1]:dim[2]],25)
            range_IQ4 = np.percentile(colonna[dim[2]:dim[3]],75) - np.percentile(colonna[dim[2]:dim[3]],25)
            lista.append((range_IQ1, range_IQ2, range_IQ3, range_IQ4))
        return lista

    def calcolo_mean_velocity(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            mean_velocity1 = np.mean(np.diff(colonna[0:dim[0]]))
            mean_velocity2 = np.mean(np.diff(colonna[dim[0]:dim[1]]))
            mean_velocity3 = np.mean(np.diff(colonna[dim[1]:dim[2]]))
            mean_velocity4 = np.mean(np.diff(colonna[dim[2]:dim[3]]))
            lista.append(mean_velocity1, mean_velocity2, mean_velocity3, mean_velocity4)
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
        return lista

    def calcolo_kurtosis(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            kurtosi1 =  sc.stats.kurtosis(colonna[0:dim[0]])
            kurtosi2 = sc.stats.kurtosis(colonna[dim[0]:dim[1]])
            kurtosi3 = sc.stats.kurtosis(colonna[dim[1]:dim[2]])
            kurtosi4 = sc.stats.kurtosis(colonna[dim[2]:dim[3]])
            lista.append((kurtosi1, kurtosi2, kurtosi3, kurtosi4))
        return lista

    def calcolo_skewness(self):
        lista = []
        for i in range(75):
            colonna = self.data[:, i]
            dim = [round(len(colonna) * 0.25), round(len(colonna) * 0.5), round(len(colonna) * 0.75), len(colonna)]
            skewness1 = sc.stats.skew(colonna[0:dim[1]])
            skewness2 = sc.stats.skew(colonna[dim[0]:dim[1]])
            skewness3 = sc.stats.skew(colonna[dim[1]:dim[2]])
            skewness4 = sc.stats.skew(colonna[dim[2]:dim[3]])
            lista.append((skewness1, skewness2, skewness3, skewness4))
        return lista


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

for data in dati_skeleton:
    valori_fin = statisticheSkeleton_finestra(data)
    #liste_std.append(valori_fin.calcolo_deviazione_std())
    #liste_mean.append(valori_fin.calcolo_media())
    #liste_max.append(valori_fin.calcolo_massimo())
    #liste_min.append(valori_fin.calcolo_minimo())
    #liste_range.append(valori_fin.calcolo_range())
    #liste_mediana.append(valori_fin.calcolo_mediana())
    #liste_MAD.append(valori_fin.calcolo_median_absolute_deviation())
    #liste_AHM.append(valori_fin.calcolo_absolute_harmonic_mean())
    #liste_rangeIQ.append(valori_fin.calcolo_range_IQ())
    #liste_mean_velocity.append(valori_fin.calcolo_mean_velocity())
    #liste_mean_energy.append(valori_fin.calcolo_mean_energy())
    #liste_kurtosi.append(valori_fin.calcolo_kurtosis())
    #liste_skew.append(valori_fin.calcolo_skewness())

#matrice_medie = np.array(liste_mean)
#np.save('matrice_3d_media.npy', matrice_medie) # questa per matrice 3d
#matrix_caricata = np.load('matrice_3d_media.npy')
#matrice_std = np.array(liste_std)
#matrice_max = np.array(liste_max)
#matrice_min = np.array(liste_min)
#matrice_range = np.array(liste_range)
#matrice_mediana = np.array(liste_mediana)
#matrice_MAD = np.array(liste_MAD)
#matrice_AHM = np.array(liste_AHM)
#matrice_rangeIQ = np.array(liste_rangeIQ)
#matrice_mean_velocity = np.array(liste_mean_velocity)
#matrice_mean_energy = np.array(liste_mean_energy)
#matrice_kurtosi = np.array(liste_kurtosi)
#matrice_skew = np.array(liste_skew)

# dopo il calcolo delle features dovrò procedere alla suddivisione di training e test


