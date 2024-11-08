import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import matplotlib
from matplotlib.animation import FuncAnimation



data_fold = "C:/Users/Giovanni Maria/Desktop/raw_npy"
extension = ".skeleton.npy"
data_list = [] # inizializzo una lista vuota
file_list = []

for filename in os.listdir(data_fold): # os.listdir mi ritorna la lista dei file della cartella adesso definisco la condizione
# prendo i file con la classe 8 e 9
    if "A008" in filename or "A009" in filename:
        file_path = os.path.join(data_fold, filename) # costruisce il percorso completo del file
        try:
            data = np.load(file_path, allow_pickle = True).item() # np.load serve per leggere dati binari specifici (in formato npy), senza item il risultato di np.load sarebbe 0, cioè contiene il dizionario come unico elemento, permette di estarre quel singolo elemento dall'array
            data_list.append(data)
            file_list.append(filename)
            #print(f"Dati caricati da {filename}:")
            #print(data)
        except Exception as e:
            print(f"Errore durante l'apertura del file {filename}:{e}")

# ANALISI QUANTITATIVA
# trovo il numero di dati skel e depth per classe

all_skel_data = []
forma_skel = []
dim_skel = []
i = 0

for data in data_list:
    skelet_data = data['skel_body0'] # come prendere un elemento dal dizionario
    file_name = file_list[i]
    forma_skel = skelet_data.shape
    dim_skel = skelet_data.size
    #print(f"File {i+1} {file_name}:")
    #print("Numero totale di elementi in skel_body:", dim_skel)
    #print("Forma di skel_body:", forma_skel)
    i += 1

def estrazione_classe(s):
    inizio = s.index("A")
    fine = inizio+4
    return s[inizio:fine]

i = 0
size_skel_8 = []
size_skel_9 = []

for data in data_list:
    substring = estrazione_classe(file_list[i])
    if substring == "A008":
        skel_8 = data["skel_body0"].size
        size_skel_8.append(skel_8)

    elif substring == "A009":
        skel_9 = data["skel_body0"].size
        size_skel_9.append(skel_9)
    i += 1

#print("Dimensioni dello skeleton in Classe 8:", size_skel_8)
#print("Dimensioni dello skeleton in Classe 9:", size_skel_9)
#print(np.mean(size_skel_8))
#print(np.mean(size_skel_9))

# lunghezza sequenza per classe
i = 0
len_skel_8 = []
len_skel_9 = []

for data in data_list:
    substring = estrazione_classe(file_list[i])
    if substring == "A008":
        skel_8 = len(data["skel_body0"])
        len_skel_8.append(skel_8)
    elif substring == "A009":
        skel_9 = len(data["skel_body0"])
        len_skel_9.append(skel_9)
    i += 1

print("Lunghezza della sequenze skeleton in Classe 8:", len_skel_8)
print("Lunghezza della sequenze skeleton in Classe 9:", len_skel_9)
print("Media lunghezza dello skeleton in Classe 8:", np.mean(len_skel_8))
print("Media della lunghezza dello skeleton in Classe 9:", np.mean(len_skel_9))


# numero di paz tot e numeri di paz per classe

cont_8 = 0
cont_9 = 0

for f in file_list:
    if "A008" in f:
        cont_8 += 1
    elif "A009" in f:
        cont_9 += 1

print(cont_8, "Num paz in classe 8")
print(cont_9, "Num paz in classe 9")


i = 0
lista_paz = []

def estrai_paz(s):
    inizio = s.index("P") # trovo l'indice di inizio
    fine = inizio + 4
    return s[inizio:fine] # estraggo la sottostringa


for stringa in file_list:
    paziente = estrai_paz(stringa)
    lista_paz.append(paziente) # creo la lista paz
    #print(f" Paziente: {paziente}")
    #print(lista_paz)

lista_ordinata = sorted(file_list, key = estrai_paz) # per ogni stringa richiamo la funzione estrai_sott, vuol dire che uso quella sottostringa per ordinare le stringhe
#print(lista_ordinata)

unique_paz = sorted(set(lista_paz))
print(unique_paz)
conteggio_paz = Counter(lista_paz) # conto le occorrenze
print(conteggio_paz)

for i in range(len(unique_paz)):
    i += 1
print("Paz tot", i)

# trovo il numero di dati per camera

ndati_cam_1 = []
ndati_cam_2 = []
ndati_cam_3 = []


for data in data_list:
    nome_file = data['file_name']
    if "C001" in nome_file:
        data1 = data['skel_body0'].size
        ndati_cam_1.append(data1)
    elif "C002" in nome_file:
        data2 = data['skel_body0'].size
        ndati_cam_2.append(data2)
    elif "C003" in nome_file:
        data3 = data['skel_body0'].size
        ndati_cam_3.append(data3)

print(f"Numero totale di dati skeleton camera 1 '{sum(ndati_cam_1)}':",ndati_cam_1)
print(f"Numero totale di dati skeleton camera 2 '{sum(ndati_cam_2)}':",ndati_cam_2)
print(f"Numero totale di dati skeleton camera 3 '{sum(ndati_cam_3)}':",ndati_cam_3)
print(f"Il numero di volte in cui camera1 registra dati skeleton è: ", len(ndati_cam_1))
print(f"Il numero di volte in cui camera2 registra dati skeleton è: ", len(ndati_cam_2))
print(f"Il numero di volte in cui camera3 registra dati skeleton è: ", len(ndati_cam_3))


# numero di volte in cui si ripetono le viste frontali di camera3, camera2

cont3_1 = 0
cont2_2 = 0
cont3_2 = 0
cont2_1 = 0
# non utilizzo camera 3 poichè essa non viene utilizzata come camera frontale

for data in data_list:
    nome_file = data['file_name']
    if "C003" in nome_file and "R001" in nome_file:
        cont3_1 += 1
    elif "C002" in nome_file and "R002" in nome_file:
        cont2_2 += 1
    elif "C003" in nome_file and "R002" in nome_file:
        cont3_2 += 1
    elif "C002" in nome_file and "R001" in nome_file:
        cont2_1 += 1

print("Numero totali di viste frontali in camera1:",cont3_1)
print("Numero totali di viste frontali in camera2:",cont2_2)
print("Numero totale di viste con camera2 come riferimento riprese da camera1:",cont3_2)
print("Numero totale di viste con camera1 come riferimento riprese da camera2:",cont2_1)

# determino il numero di setting e la frequenza di ogni setting

setting = []

def estr_set(s):
    inizio = s.index("S")
    fine = inizio + 4
    return s[inizio:fine]

for data in data_list:
    substring = estr_set(data['file_name'])
    setting.append(substring)

setting = sorted(set(setting))

cont_tot_sett = []
cont = 0
i = 0
for i in range(len(setting)):
    for data in data_list:
        substring = estr_set(data['file_name'])
        if substring == setting[i]:
            cont += 1
    cont_tot_sett.append(cont)
    i += 1
    cont = 0

i = 0
for sett in cont_tot_sett:
    print(f"Il numero di {setting[i]}:",sett)
    i += 1

# numero dati per setting

def num_dati_setting(setting):
    lista = []
    for data in data_list:
        if setting in data['file_name']:
            dati = data['skel_body0'].size
            lista.append(dati)
    return sum(lista), np.mean(lista)


num_totdati_setting = []
mean_dati_setting = []
for i in range(len(setting)):
    [num_dati, media_dati] = num_dati_setting(setting[i])
    print(f"Numero dati totali {setting[i]}: {num_dati}, media dati per sequenza:", media_dati)
    num_totdati_setting.append(num_dati)
    mean_dati_setting.append(media_dati)

# numero di setting per camera

def camera_sett(cam,data_list,setting):
    cont = 0
    index = 0
    index_tot = []
    cont_tot = []
    for i in range(len(setting)):
        cont = 0
        for data in data_list:
            substring = estr_set(data['file_name'])
            if cam in data['file_name']:
                if substring == setting[i]:
                    cont += 1
                    index = i
        cont_tot.append(cont)
        index_tot.append(index)
    return cont_tot,index_tot

camere = ["C001","C002","C003"]
[sett_camera1,idx] = camera_sett(camere[0],data_list,setting)
for i in range(len(sett_camera1)):
    print(f"Il {setting[i]} di CAM1 viene effettuato:",sett_camera1[i])

[sett_camera2,idx] = camera_sett(camere[1],data_list,setting)
for i in range(len(sett_camera2)):
    print(f"Il {setting[i]} di CAM2 viene effettuato:",sett_camera2[i])

[sett_camera3,idx] = camera_sett(camere[2],data_list,setting)
for i in range(len(idx)):
    print(f"Il {setting[i]} di CAM3 viene effettuato:",sett_camera3[i])

# lunghezza sequenze per ogni camera

def lungh_seq(cam,data_list):
    leng_tot = []
    lungh_seq = 0
    for data in data_list:
        if cam in data['file_name']:
            lungh_seq = len(data['skel_body0'])
            leng_tot.append(lungh_seq)
    return leng_tot

len_seq1 = lungh_seq(camere[0],data_list)
print(f"Media Lunghezza sequenze {camere[0]}: {(np.mean(len_seq1))} Lunghezza sequenze {camere[0]}:",len_seq1)

len_seq2 = lungh_seq(camere[1],data_list)
print(f"Media Lunghezza sequenze {camere[1]}: {(np.mean(len_seq2))} Lunghezza sequenze {camere[1]}:",len_seq2)

len_seq3 = lungh_seq(camere[2],data_list)
print(f"Media Lunghezza sequenze {camere[2]}: {(np.mean(len_seq3))} Lunghezza sequenze {camere[2]}:",len_seq3)


# determino la lunghezza sequenze per ogni setting

def len_seq_set(setting,data_list):

    lista = []
    for data in data_list:
        if setting in data['file_name']:
            lungh_seq = len(data['skel_body0'])
            lista.append(lungh_seq)

    return lista

mean_seq_set = []
for i in range(len(setting)):
    len_seq_setting = len_seq_set(setting[i],data_list)
    print(f"Media Lunghezza sequenze {np.mean(len_seq_setting)} Lunghezza sequenze{setting[i]}:",len_seq_setting)
    mean_seq_set.append(np.mean(len_seq_setting))

def len_seq_paz(pazienti,data_list):

    lista = []
    for data in data_list:
        if pazienti in data['file_name']:
            lungh_seq = len(data['skel_body0'])
            lista.append(lungh_seq)

    return lista

mean_seq_paz = []
for i in range(len(unique_paz)):
    len_seq_paziente = len_seq_paz(unique_paz[i],data_list)
    print(f"Media Lunghezza sequenze {np.mean(len_seq_paziente)} Lunghezza sequenze{unique_paz[i]}:", len_seq_paziente)
    mean_seq_paz.append(np.mean(len_seq_paziente))


# numero di elementi per classe per paz

def num_dati_paz_class(classe,pazienti,data_list):

    lista = []
    for data in data_list:
        if classe in data['file_name']:
            if pazienti in data['file_name']:
                num_dati = data['skel_body0'].size
                lista.append(num_dati)

    return lista

num_tot_dati_paz_classe8 = []
media_dati_paz_classe8 = []
i = 0
for i in range(len(unique_paz)):
    size_dati = num_dati_paz_class("A008",unique_paz[i],data_list)
    print(f"Numeri dati totali {unique_paz[i]} in classe 8: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_paz_classe8.append(sum(size_dati))
    media_dati_paz_classe8.append(np.mean(size_dati))


num_tot_dati_paz_classe9 = []
media_dati_paz_classe9 = []
i = 0
for i in range(len(unique_paz)):
    size_dati = num_dati_paz_class("A009", unique_paz[i], data_list)
    print(f"Numeri dati totali {unique_paz[i]} in classe 9: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_paz_classe9.append(sum(size_dati))
    media_dati_paz_classe9.append(np.mean(size_dati))

# numero di elementi per classe per camera

def num_dati_camera_class(classe,camera,data_list):

    lista = []
    for data in data_list:
        if classe in data['file_name']:
            if camera in data['file_name']:
                num_dati = data['skel_body0'].size
                lista.append(num_dati)

    return lista

num_tot_dati_cam_classe8 = []
media_dati_cam_classe8 = []
i = 0
for i in range(len(camere)):
    size_dati = num_dati_paz_class("A008",camere[i],data_list)
    print(f"Numeri dati totali {camere[i]} in classe 8: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam_classe8.append(sum(size_dati))
    media_dati_cam_classe8.append(np.mean(size_dati))

num_tot_dati_cam_classe9 = []
media_dati_cam_classe9 = []
i = 0
for i in range(len(camere)):
    size_dati = num_dati_paz_class("A009", camere[i], data_list)
    print(f"Numeri dati totali {camere[i]} in classe 9: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam_classe9.append(sum(size_dati))
    media_dati_cam_classe9.append(np.mean(size_dati))

# numero di elementi per camera e per ripetizione

def num_dati_camera_class(camera,ripetizione,data_list):

    lista = []
    for data in data_list:
        if camera in data['file_name']:
            if ripetizione in data['file_name']:
                num_dati = data['skel_body0'].size
                lista.append(num_dati)

    return lista

ripetizioni = ["R001","R002"]
num_tot_dati_cam1_rip = []
media_dati_cam1_rip = []
i = 0
for i in range(len(ripetizioni)):
    size_dati = num_dati_paz_class("C001",ripetizioni[i],data_list)
    print(f"Numeri dati totali C001 in {ripetizioni[i]}: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam1_rip.append(sum(size_dati))
    media_dati_cam1_rip.append(np.mean(size_dati))

num_tot_dati_cam2_rip = []
media_dati_cam2_rip = []
i = 0
for i in range(len(ripetizioni)):
    size_dati = num_dati_paz_class("C002",ripetizioni[i],data_list)
    print(f"Numeri dati totali C002 in {ripetizioni[i]}: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam2_rip.append(sum(size_dati))
    media_dati_cam2_rip.append(np.mean(size_dati))

num_tot_dati_cam3_rip = []
media_dati_cam3_rip = []
i = 0
for i in range(len(ripetizioni)):
    size_dati = num_dati_paz_class("C003",ripetizioni[i],data_list)
    print(f"Numeri dati totali C003 in {ripetizioni[i]}: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam3_rip.append(sum(size_dati))
    media_dati_cam3_rip.append(np.mean(size_dati))

# numero di elementi per classe e ripetizione

def num_dati_ripetizioni_class(classe,ripetizione,data_list):

    lista = []
    for data in data_list:
        if classe in data['file_name']:
            if ripetizione in data['file_name']:
                num_dati = data['skel_body0'].size
                lista.append(num_dati)

    return lista

ripetizioni = ["R001","R002"]
num_tot_dati_classe8_rip = []
media_dati_classe9_rip = []
i = 0
for i in range(len(ripetizioni)):
    size_dati = num_dati_paz_class("A008",ripetizioni[i],data_list)
    print(f"Numeri dati totali A008 in {ripetizioni[i]}: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam1_rip.append(sum(size_dati))
    media_dati_cam1_rip.append(np.mean(size_dati))

num_tot_dati_classe9_rip = []
media_dati_classe9_rip = []
i = 0
for i in range(len(ripetizioni)):
    size_dati = num_dati_paz_class("A009",ripetizioni[i],data_list)
    print(f"Numeri dati totali A009 in {ripetizioni[i]}: {sum(size_dati)}, Media dati:", np.mean(size_dati))
    num_tot_dati_cam2_rip.append(sum(size_dati))
    media_dati_cam2_rip.append(np.mean(size_dati))


# lunghezza sequenze viste frontali

def len_seq_viste_frontali(camera,ripetizione):
    lista = []
    lista1 = []
    for data in data_list:
        if camera in data['file_name']:
            if ripetizione in data['file_name']:
                len_seq = len(data['skel_body0'])
                lista.append(len_seq)
                num_dati = data['skel_body0'].size
                lista1.append(num_dati)

    return lista,lista1

len_seq_vf3 = []
num_dati_vf3 = []
len_seq_vf2 = []
num_dati_vf2 = []
[len_seq_vf3, num_dati_vf3] = len_seq_viste_frontali("C003","R001")
[len_seq_vf2, num_dati_vf2] = len_seq_viste_frontali("C002","R002")
print("Sequenza temporale vista frontale camera 3", len_seq_vf3)
print("Sequenza temporale vista frontale camera 2", len_seq_vf2)
print("Numero dati vista frontale camera 3", num_dati_vf3)
print("Numero dati vista frontale camera 2", num_dati_vf2)

# frequenza setting per viste frontali

def camera_sett(cam,ripetizione,setting):
    cont = 0
    index = 0
    index_tot = []
    cont_tot = []
    for i in range(len(setting)):
        cont = 0
        for data in data_list:
            if cam in data['file_name']:
                if setting[i] in data['file_name']:
                    if ripetizione in data['file_name']:
                        cont += 1
                        index = i
        cont_tot.append(cont)
        index_tot.append(index)
    return cont_tot,index_tot

camere = ["C002","C003"]
[sett_camera_vis_fro2,idx] = camera_sett(camere[0],"R002",setting)
for i in range(len(sett_camera2)):
    print(f"Il {setting[i]} di CAM2 viene effettuato:",sett_camera_vis_fro2[i])

[sett_camera_vis_fro3,idx] = camera_sett(camere[1],"R001",setting)
for i in range(len(sett_camera3)):
    print(f"Il {setting[i]} di CAM3 viene effettuato:",sett_camera_vis_fro3[i])

# analisi paz-ripetizioni

def paziente_ripetizione(ripetizione,classe,paziente):

    cont = 0
    for data in data_list:
        if ripetizione in data['file_name']:
            if classe in data['file_name']:
                if paziente in data['file_name']:
                    cont += 1

    return cont

paziente_ripetizione1_8 = []
paziente_ripetizione1_9 = []
for i in range(len(unique_paz)):
    paz_rip1 = paziente_ripetizione("R001","A008",unique_paz[i])
    print(f" Numero di ripetizioni fatti dal {unique_paz[i]}:",paz_rip1)
    paziente_ripetizione1_8.append(paz_rip1)

for i in range(len(unique_paz)):
    paz_rip1 = paziente_ripetizione("R001", "A009", unique_paz[i])
    print(f" Numero di ripetizioni fatti dal {unique_paz[i]}:", paz_rip1)
    paziente_ripetizione1_9.append(paz_rip1)

paziente_ripetizione2_8 = []
paziente_ripetizione2_9 = []
for i in range(len(unique_paz)):
    paz_rip1 = paziente_ripetizione("R002","A008",unique_paz[i])
    print(f" Numero di ripetizioni fatti dal {unique_paz[i]}:",paz_rip1)
    paziente_ripetizione2_8.append(paz_rip1)

for i in range(len(unique_paz)):
    paz_rip1 = paziente_ripetizione("R002", "A009", unique_paz[i])
    print(f" Numero di ripetizioni fatti dal {unique_paz[i]}:", paz_rip1)
    paziente_ripetizione2_9.append(paz_rip1)

# numero paz viste frontali e lunghezza sequenze viste frontali

def paziente_vista(ripetizione,classe,paziente):

    lista = []
    media_seq = 0
    cont = 0
    for data in data_list:
        if ripetizione in data['file_name']:
            if classe in data['file_name']:
                if paziente in data['file_name']:
                    cont += 1
                    seq_len = len(data['skel_body0'])
                    lista.append(seq_len)

    return cont, np.mean(lista)

media_seq_paz_vista3 = []
paziente_vista3 = []
for i in range(len(unique_paz)):
    [paz_vista3, media_vista3] = paziente_vista("R001","C003",unique_paz[i])
    print(f"Paziente {unique_paz[i]} in vista frontale 3:",paz_vista3, media_vista3)
    paziente_vista3.append(paz_vista3)
    media_seq_paz_vista3.append(media_vista3)

media_seq_paz_vista2 = []
paziente_vista2 = []
for i in range(len(unique_paz)):
    [paz_vista2, media_vista2] = paziente_vista("R002","C002",unique_paz[i])
    print(f"Paziente {unique_paz[i]} in vista frontale 2:",paz_vista2, media_vista2)
    paziente_vista2.append(paz_vista2)
    media_seq_paz_vista2.append(media_vista2)

# determinare i setting per paziente

def paziente_setting(paziente,setting):
    lista = []
    cont = 0

    for i in range(len(setting)):
        set = setting[i]
        for data in data_list:
            if paziente in data['file_name']:
                if set in data['file_name']:
                    cont += 1

        lista.append(cont)
        cont = 0
    return lista

lista_paz_setting = []
for paz in unique_paz:
    paz_sett = paziente_setting(paz,setting)
    print(f"I setting del {paz}:",paz_sett)
    lista_paz_setting.append(paz_sett)

# determinare i setting per paziente in vista frontale

def paziente_setting(paziente,setting,camere,ripetizione):
    lista = []
    cont = 0

    for i in range(len(setting)):
        set = setting[i]
        for data in data_list:
            if paziente in data['file_name']:
                if set in data['file_name']:
                    if camere in data['file_name']:
                        if ripetizione in data['file_name']:
                            cont += 1

        lista.append(cont)
        cont = 0
    return lista

lista_paz_setting_vf2 = []
lista_paz_setting_vf3 = []

for paz in unique_paz:
    paz_sett_vf2 = paziente_setting(paz,setting,camere[0],"R002")
    print(f"I setting del {paz} in vista frontale 2:",paz_sett_vf2)
    lista_paz_setting_vf2.append(paz_sett_vf2)

for paz in unique_paz:
    paz_sett_vf3 = paziente_setting(paz, setting, camere[1],"R001")
    print(f"I setting del {paz} in vista frontale 3:", paz_sett_vf3)
    lista_paz_setting_vf3.append(paz_sett_vf3)

# determinare i setting per paziente nelle due classi

def paziente_setting(paziente,setting,classe):
    lista = []
    cont = 0

    for i in range(len(setting)):
        set = setting[i]
        for data in data_list:
            if paziente in data['file_name']:
                if set in data['file_name']:
                    if classe in data['file_name']:
                        cont += 1

        lista.append(cont)
        cont = 0
    return lista

lista_paz_setting_cl8 = []
lista_paz_setting_cl9 = []

for paz in unique_paz:
    paz_sett_cl8 = paziente_setting(paz,setting,"A008")
    print(f"I setting del {paz} nella classe sit down:",paz_sett_cl8)
    lista_paz_setting_cl8.append(paz_sett_cl8)

for paz in unique_paz:
    paz_sett_cl9 = paziente_setting(paz, setting, "A009")
    print(f"I setting del {paz} nella classe stand up:", paz_sett_cl9)
    lista_paz_setting_cl9.append(paz_sett_cl9)

# determinare i setting per paziente nelle due classi in vista frontale

def paziente_setting(paziente,setting,classe,camera,ripetizione):
    lista = []
    cont = 0

    for i in range(len(setting)):
        set = setting[i]
        for data in data_list:
            if paziente in data['file_name']:
                if set in data['file_name']:
                    if classe in data['file_name']:
                        if camera in data['file_name']:
                            if ripetizione in data['file_name']:
                                cont += 1

        lista.append(cont)
        cont = 0
    return lista

lista_paz_setting_cl8_vf2 = []
lista_paz_setting_cl8_vf3 = []
lista_paz_setting_cl9_vf2 = []
lista_paz_setting_cl9_vf3 = []

for paz in unique_paz:
    paz_sett_cl8_vf2 = paziente_setting(paz,setting,"A008",camere[0],"R002")
    print(f"I setting del {paz} nella classe sit down in vista frontale 2:",paz_sett_cl8_vf2)
    lista_paz_setting_cl8_vf2.append(paz_sett_cl8_vf2)

for paz in unique_paz:
    paz_sett_cl8_vf3 = paziente_setting(paz,setting,"A008",camere[1],"R001")
    print(f"I setting del {paz} nella classe sit down in vista frontale 3:",paz_sett_cl8_vf3)
    lista_paz_setting_cl8_vf3.append(paz_sett_cl8_vf3)

for paz in unique_paz:
    paz_sett_cl9_vf2 = paziente_setting(paz, setting, "A009",camere[0],"R002")
    print(f"I setting del {paz} nella classe stand up in vista frontale 2:", paz_sett_cl9_vf2)
    lista_paz_setting_cl9_vf2.append(paz_sett_cl9_vf2)

for paz in unique_paz:
    paz_sett_cl9_vf3 = paziente_setting(paz, setting, "A009",camere[1],"R001")
    print(f"I setting del {paz} nella classe stand up in vista frontale 3:", paz_sett_cl9_vf3)
    lista_paz_setting_cl9_vf3.append(paz_sett_cl9_vf3)


# ANALISI QUALITATIVA

# diagramma a torta per vedere se c'è un bilanciamento tra le due classi

plt.style.use("ggplot")
labels = ['Sit Down','Stand Up']
colors = ['red','blue']
slices = [cont_8, cont_9]
plt.pie(slices, labels = labels, colors = colors,autopct = lambda x: f'{x:.1f}%')
plt.title("Suddivisone di pazienti in classi Sit Down e Stand Up")
#plt.show()

# istogramma per vedere la lunghezza delle sequenze tra le due classi

fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.hist(len_skel_8, bins = 10)
ax1.set_title("Lunghezza Sequenze")
ax1.set_xlabel('Sit Down')

ax2.hist(len_skel_9, bins = 10)
ax2.set_xlabel('Stand Up')
#plt.show()

# diagramma a torta per analizzare il numero di dati skeleton in CAM1,CAM2,CAM3

labels = ['CAM1','CAM2','CAM3']
colors = ['red','blue','yellow']
slices = [sum(ndati_cam_1),sum(ndati_cam_2),sum(ndati_cam_3)]

plt.pie(slices, labels = labels, colors = colors,autopct = lambda x: f'{x:.1f}%')
plt.title("Percentuali dati nella varie camere desiderate")
#plt.show()

# istogramma per analizzare la lunghezza delle sequenze delle varie camere

fig, (ax1,ax2,ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = True)

ax1.hist(len_seq1, bins = 5)
ax1.set_title("Lunghezza Sequenze")
ax1.set_ylabel('CAM1')

ax2.hist(len_seq2, bins = 5)
ax2.set_ylabel('CAM2')

ax3.hist(len_seq3, bins = 5)
ax3.set_ylabel('CAM3')
#plt.show()

# Visualizzazione del numero di viste frontali in CAM3 e CAM2, nonchè il numero di ripetizioni

keys_cam = ["CAM3","CAM2"]
values = [cont3_1, cont2_2]

plt.bar(keys_cam, values)
plt.title('Numero viste frontali')
plt.ylabel('Frequenza')
#plt.show()

values = [cont3_2, cont2_1]

plt.bar(keys_cam, values)
plt.title('Numero viste')
plt.ylabel('Frequenza')
#plt.show()

# media lunghezza sequenze per ogni paziente

keys_paz = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40']
plt.bar(keys_paz,mean_seq_paz)
plt.title('Lunghezza Media Sequenze per Paziente')
plt.ylabel('Frequenza')
#plt.show()

# media lunghezza sequenze per ogni setting

keys_set = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
plt.bar(keys_set,mean_seq_set)
plt.title('Lunghezza Media Sequenza per Setting')
plt.ylabel('Frequenza')
plt.xlabel('Setting')
#plt.show()

# numero di setting

plt.bar(keys_set,cont_tot_sett)
plt.title('Frequenza Setting')
plt.ylabel('Frequenza')
plt.xlabel('Setting')
#plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = True)

ax1.bar(keys_set, sett_camera1)
ax1.set_title("Frequenza Setting Camere")
ax1.set_ylabel('Set Cam1')

ax2.bar(keys_set, sett_camera2)
ax2.set_ylabel('Set Cam2')

ax3.bar(keys_set, sett_camera3)
ax3.set_ylabel('Set Cam3')
ax3.set_xlabel('Setting')
#plt.show()

# lunghezza sequenze vista frontale 3 e 2

fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.hist(len_seq_vf3, bins = 5)
ax1.set_title("Lunghezza Sequenze viste frontali")
ax1.set_ylabel('Vista Frontale 3')

ax2.hist(len_seq_vf2, bins = 5)
ax2.set_ylabel('Vista Frontale 2')
#plt.show()

fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.hist(num_dati_vf3, bins = 5)
ax1.set_title("Numero dati viste frontali")
ax1.set_ylabel('Vista Frontale 3')

ax2.hist(num_dati_vf2, bins = 5)
ax2.set_ylabel('Vista Frontale 2')
#plt.show()

# numero totali dati paz per classe

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.bar(keys_paz, num_tot_dati_paz_classe8)
ax1.set_title("Numero dati Pazienti-Classe")
ax1.yaxis.set_label_position("right")
ax1.set_ylabel('Sit Down')

ax2.bar(keys_paz, num_tot_dati_paz_classe9)
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Stand Up')
#plt.show()

# numero totali dati camere classe

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
keys_camere = ["C001","C002","C003"]

ax1.bar(keys_camere, media_dati_cam_classe8)
ax1.set_title("Media dati Camere-Classe per sequenza")
ax1.set_ylabel('Sit Down')

ax2.bar(keys_camere, media_dati_cam_classe9)
ax2.set_ylabel('Stand Up')
#plt.show()

# numero media dati setting

plt.bar(keys_set, mean_dati_setting)
plt.title("Media dati Setting per sequenza")
plt.xlabel("Setting")
#plt.show()

# frequenza setting viste frontali

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.bar(keys_set, sett_camera_vis_fro3)
ax1.set_title("Frequenza Setting viste frontali")
ax1.set_ylabel('Vista Frontale 3')

ax2.bar(keys_set, sett_camera_vis_fro2)
ax2.set_ylabel('Vista Frontale 2')
#plt.show()

# freq paz in vista frontale e media seq temporale

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.bar(keys_paz,  media_seq_paz_vista3)
ax1.set_title("Media sequenze temporali Pazienti")
ax1.set_ylabel('Vista Frontale 3')

ax2.bar(keys_paz, media_seq_paz_vista2)
ax2.set_ylabel('Vista Frontale 2')
#plt.show()

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax1.bar(keys_paz,paziente_vista3)
ax1.set_title("Frquenza Pazienti")
ax1.set_ylabel('Vista Frontale 3')

ax2.bar(keys_paz,paziente_vista2)
ax2.set_ylabel('Vista Frontale 2')
#plt.show()

# determinare i setting per paziente in vista frontale

data_sett_paz = np.array(lista_paz_setting)

# grafico a linee

for i, paz in enumerate(keys_paz):
    plt.plot(keys_set, data_sett_paz[i], marker = 'o', label = keys_paz[i])

plt.title("Grafico a Linee")
plt.xlabel("Setting")
plt.ylabel("Frequenza")
plt.legend(title = "Pazienti")

#plt.show()

# scatter plotòl
plt.figure(figsize= (10,6))

for i, frequenze in enumerate(lista_paz_setting):
    plt.scatter(keys_set, frequenze,label = keys_paz[i])

plt.title("Scatter Plot del Setting per Pazienti")
plt.xlabel("Setting")
plt.ylabel("Frequenza")
plt.legend(title = "Pazienti")
#plt.show()


for i in range(len(keys_paz)):
    plt.bar(keys_set, data_sett_paz[i], bottom = np.sum(data_sett_paz[:i], axis = 0), label = keys_paz[i])
    plt.title("Grafico a Barre (Stacked)")
    plt.xlabel("Setting")
    plt.ylabel("Frequenza")
    plt.legend(title = "Pazienti")


for data in data_list:
    if 'skel_body0' in data:
        skel_body0 = data['skel_body0']
        skel_body_reshape = skel_body0.reshape(skel_body0.shape[0], -1)
        data['skel_body0'] = skel_body_reshape

connections = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (20, 8), (6, 7), (8, 9), (9, 10),
               (10, 11), (11, 23), (11, 24), (7, 22), (0, 12), (12, 13), (13, 14), (14, 15), (16, 17),
               (17, 18), (18, 19), (7, 21), (0, 16)]
dt = 1 / 30  # Freq Camp

def show_skeleton(positions, connection, dt, title=None, slowing_parameter=1):
    plt.close('all')
    # Extract x, y, z coordinates of joints
    x = positions[:, 0::3]
    max_x = np.max(x)
    min_x = np.min(x)
    y = positions[:, 1::3]
    max_y = np.max(y)
    min_y = np.min(y)
    z = positions[:, 2::3]
    max_z = np.max(z)
    min_z = np.min(z)

    # Plot each frame
    matplotlib.use("TkAgg")
    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    ax.set_aspect("auto")
    ax.view_init(elev=90, azim=-90)

    for frame in range(positions.shape[0]):
        # Initialize axis
        plt.cla()
        ax.axis("off")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        ax.set_title(title)

        frame_x = x[frame]     #[frame, :, 0]
        frame_y = y[frame]     #[frame, :, 1]
        frame_z = z[frame]     #[frame, :, 2]

        # Plot connections
        for connection in connections:
                joint1_pos = (frame_x[connection[0]], frame_y[connection[0]], frame_z[connection[0]])
                joint2_pos = (frame_x[connection[1]], frame_y[connection[1]], frame_z[connection[1]])
                ax.plot([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]], [joint1_pos[2], joint2_pos[2]],
                        c="gray")

        # Plot joints
        color = "gray"
        ax.scatter(frame_x, frame_y, frame_z, color=color, marker="o", linewidth=0, alpha=1)

        plt.draw()
        plt.pause(dt * slowing_parameter)
    plt.ioff()
    plt.show()

for data in data_list:
    if "P001" in data['file_name']:
        if "A009" in data['file_name']:
            if "C003" in data['file_name']:
                if "R002" in data['file_name']:
                    positions = data['skel_body0']

#show_skeleton(positions, connections, dt)

#dizionario = {}
#for data in data_list:
    #nome = data['file_name']
    #valore = data['skel_body0']
    #dizionario[nome] = valore

#with open('dizionario.pkl', 'wb') as file:
    #pickle.dump(dizionario, file)

def show_skeleton(positions, connections, dt, title=None, slowing_parameter=1):
    output_filename = "skeleton_animation.gif"

    if os.path.exists(output_filename):
        os.remove(output_filename)
    # Estraggo le coordinate X, Y, Z delle articolazioni
    x = positions[:, 0::3]
    y = positions[:, 1::3]
    z = positions[:, 2::3]

    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    ax.set_box_aspect([1, 1, 1])

    ax.view_init(elev = 90, azim = -90)

    def update(frame):
        ax.cla()  # Pulisce l'asse a ogni frame
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        if title:
            ax.set_title(title)
        ax.axis("off")

        # Plot skeleton connections
        for connection in connections:
            joint1_pos = (x[frame, connection[0]], y[frame, connection[0]], z[frame, connection[0]])
            joint2_pos = (x[frame, connection[1]], y[frame, connection[1]], z[frame, connection[1]])
            ax.plot([joint1_pos[0], joint2_pos[0]],
                    [joint1_pos[1], joint2_pos[1]],
                    [joint1_pos[2], joint2_pos[2]],
                    color="gray")

        # Plot joints
        ax.scatter(x[frame], y[frame], z[frame], color="blue", marker="o", s=10, alpha=0.7)

    # Uso FuncAnimation per generare l'animazione
    ani = FuncAnimation(fig, update, frames=positions.shape[0], interval=dt * 1000 * slowing_parameter)

    # Salvo l'animazione come file gif
    ani.save(output_filename, writer="imagemagick")
    plt.close()

show_skeleton(positions, connections, dt=1/30)