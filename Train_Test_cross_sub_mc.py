import numpy as np
import pickle
import random

with open('nome_file_vf2.pkl','rb') as f:
    nome_file_vf2 = pickle.load(f)

matrice_feat_mc = np.load('matrice_features_mc.npy')

dim_train = []
dim_test = []
# faccio una suddivisione cross subject
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

    pazienti_train = random.sample(lista_paz, k = 75)

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

    nome_file_vf2_train = []
    for i in index_train:
        nome = nome_file_vf2[i]
        nome_file_vf2_train.append(nome)

    nome_file_vf2_test = []
    for i in index_test:
        nome = nome_file_vf2[i]
        nome_file_vf2_test.append(nome)

    data_train_vf2 = matrice_feat_mc[index_train,:]
    data_test_vf2 = matrice_feat_mc[index_test,:]

    lung_train = len(data_train_vf2)
    lung_test = len(data_test_vf2)

    dim_train.append(lung_train)
    dim_test.append(lung_test)


print(np.sort(dim_train))
print(np.sort(dim_test))

#np.save('matrice_features_train_cs.npy', data_train_vf2)
#np.save('matrice_features_test_cs.npy', data_test_vf2)

#with open('nome_file_vf2_train_cs.pkl', 'wb') as f:
#    pickle.dump(nome_file_vf2_train, f)

#with open('nome_file_vf2_test_cs.pkl', 'wb') as f:
    #pickle.dump(nome_file_vf2_test, f)

