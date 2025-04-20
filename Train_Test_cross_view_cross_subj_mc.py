import numpy as np
import pickle
import random

with open('nome_file_vf2.pkl','rb') as f:
    nome_file_vf2 = pickle.load(f)

matrice_feat_mc = np.load('matrice_features_mc.npy')

# faccio una suddivisione cross subject
dim_train = []
dim_test = []

for i in range(100):

    def estrazione_paz(stringa):
        inizio = stringa.index("P")
        fine = inizio + 4
        return stringa[inizio:fine]

    pazienti = []
    for nome in nome_file_vf2:
        substring = estrazione_paz(nome)
        pazienti.append(substring)

    lista_paz = sorted(set(pazienti))

    pazienti_train = random.sample(lista_paz, k = 70)

    pazienti_test = []

    for paz in lista_paz:
        if paz not in pazienti_train:
            pazienti_test.append(paz)

    pazienti_train = sorted(pazienti_train)
    pazienti_test = sorted(pazienti_test)


    index_train = []
    for paz in pazienti_train:
        for index, nome in enumerate(nome_file_vf2):
            if paz in nome:
                index_train.append(index)

    index_test = []
    for paz in pazienti_test:
        for index, nome in enumerate(nome_file_vf2):
            if paz in nome:
                index_test.append(index)

    nome_file_vf2_train_cs = []
    for i in index_train:
        nome = nome_file_vf2[i]
        nome_file_vf2_train_cs.append(nome)

    nome_file_vf2_test_cs = []
    for i in index_test:
        nome = nome_file_vf2[i]
        nome_file_vf2_test_cs.append(nome)

    data_train_vf2_cs = matrice_feat_mc[index_train,:]
    data_test_vf2_cs = matrice_feat_mc[index_test,:]


# faccio una suddivisione cross view

    def estrazione_set(stringa):
        inizio = stringa.index("S")
        fine = inizio + 4
        return stringa[inizio:fine]

    setting = []
    for nome in nome_file_vf2:
        substring = estrazione_set(nome)
        setting.append(substring)

    lista_set = sorted(set(setting))

    set_train = random.sample(lista_set, k = 20)

    set_test = []

    for setting in lista_set:
        if setting not in set_train:
            set_test.append(setting)

    set_train = sorted(set_train)
    set_test = sorted(set_test)

    indici_train_cs_cv = []
    indici_test_cs_cv = []

    for setting in set_train:
        for index, nome in enumerate(nome_file_vf2_train_cs):
            if setting in nome:
                indici_train_cs_cv.append(index)

    for setting in set_test:
        for index, nome in enumerate(nome_file_vf2_test_cs):
            if setting in nome:
                indici_test_cs_cv.append(index)


    data_train_vf2_cs_cv = matrice_feat_mc[indici_train_cs_cv,:]
    data_test_vf2_cs_cv = matrice_feat_mc[indici_test_cs_cv,:]


    nome_file_vf2_train_cs_cv = []
    for i in indici_train_cs_cv:
        nome = nome_file_vf2_train_cs[i]
        nome_file_vf2_train_cs_cv.append(nome)


    nome_file_vf2_test_cs_cv = []
    for i in indici_test_cs_cv:
        nome = nome_file_vf2_test_cs[i]
        nome_file_vf2_test_cs_cv.append(nome)

    lung_train = len(data_train_vf2_cs_cv)
    lung_test = len(data_test_vf2_cs_cv)

    dim_train.append(lung_train)
    dim_test.append(lung_test)

print(np.sort(dim_train))
print(np.sort(dim_test))

#np.save('matrice_features_train_cs_cv.npy', data_train_vf2_cs_cv)
#np.save('matrice_features_test_cs_cv.npy', data_test_vf2_cs_cv)

#with open('nome_file_vf2_train_cs_cv.pkl', 'wb') as f:
#    pickle.dump(nome_file_vf2_train_cs_cv, f)

#with open('nome_file_vf2_test_cs_cv.pkl', 'wb') as f:
    #pickle.dump(nome_file_vf2_test_cs_cv, f)

