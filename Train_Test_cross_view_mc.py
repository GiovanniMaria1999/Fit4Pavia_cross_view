import numpy as np
import pickle
import random

with open('nome_file_vf2.pkl','rb') as f:
    nome_file_vf2 = pickle.load(f)

matrice_feat_mc = np.load('matrice_features_mc.npy')

indici = []
for index, value in enumerate(nome_file_vf2):
    indici.append(index)

dim_train = []
dim_test = []
# faccio una suddivisione cross setting (cross view)

for i in range(100):

    def estrazione_set(stringa):
        inizio = stringa.index("S")
        fine = inizio + 4
        return stringa[inizio:fine]

    setting = []
    for nome in nome_file_vf2:
        substring = estrazione_set(nome)
        setting.append(substring)

    lista_set = sorted(set(setting))

    set_train = random.sample(lista_set, k = 23)

    set_test = []

    for setting in lista_set:
        if setting not in set_train:
            set_test.append(setting)

    set_train = sorted(set_train)
    set_test = sorted(set_test)

    indici_train = []
    indici_test = []

    for setting in set_train:
        for index, nome in enumerate(nome_file_vf2):
            if setting in nome:
                indici_train.append(index)

    for setting in set_test:
        for index, nome in enumerate(nome_file_vf2):
            if setting in nome:
                indici_test.append(index)


    nome_file_train_vf2 = []
    nome_file_test_vf2 = []

    for i in indici_train:
        nome = nome_file_vf2[i]
        nome_file_train_vf2.append(nome)

    for i in indici_test:
        nome = nome_file_vf2[i]
        nome_file_test_vf2.append(nome)


    data_train_vf2 = matrice_feat_mc[indici_train,:]
    data_test_vf2 = matrice_feat_mc[indici_test,:]

    lung_train = len(data_train_vf2)
    lung_test = len(data_test_vf2)

    dim_train.append(lung_train)
    dim_test.append(lung_test)


print(np.sort(dim_train))
print(np.sort(dim_test))


#np.save('matrice_features_train_cv.npy', data_train_vf2)
#np.save('matrice_features_test_cv.npy', data_test_vf2)

#with open('nome_file_vf2_train_cv.pkl', 'wb') as f:
#    pickle.dump(nome_file_train_vf2, f)

#with open('nome_file_vf2_test_cv.pkl', 'wb') as f:
    #pickle.dump(nome_file_test_vf2, f)