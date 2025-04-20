import numpy as np
import pickle
import random

with open('nome_file_vf2.pkl','rb') as f:
    nome_file_vf2 = pickle.load(f)

matrice_feat_mc = np.load('matrice_features_mc.npy')

indici = []
for index, value in enumerate(nome_file_vf2):
    indici.append(index)

# faccio una suddivisione non cross subj (70/30)

dim_train = round(70*len(indici)/100)
indici_train = random.sample(indici, dim_train)

indici_test = []
for i in indici:
    if i not in indici_train:
        indici_test.append(i)

nome_file_vf2_train = []
for i in indici_train:
    nome = nome_file_vf2[i]
    nome_file_vf2_train.append(nome)

nome_file_vf2_test = []
for i in indici_test:
    nome = nome_file_vf2[i]
    nome_file_vf2_test.append(nome)


data_train_vf2 = matrice_feat_mc[indici_train,:]
data_test_vf2 = matrice_feat_mc[indici_test,:]


#np.save('matrice_features_train_ncs.npy', data_train_vf2)
#np.save('matrice_features_test_ncs.npy', data_test_vf2)

#with open('nome_file_vf2_train_ncs.pkl', 'wb') as f:
#    pickle.dump(nome_file_vf2_train, f)

#with open('nome_file_vf2_test_ncs.pkl', 'wb') as f:
    #pickle.dump(nome_file_vf2_test, f)







