import pickle
import random

with open('dizionario_mc_vf2.pkl', 'rb') as file:
    dizionario = pickle.load(file)


dati_nome_chiave = list(dizionario.keys())
dati_skeleton_body = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste

lista_indici = []
for i in range(len(dati_skeleton_body)):
    lista_indici.append(i)


index_test = []

dim_train = round(70*len(dati_skeleton_body)/100)
index_train = random.sample(lista_indici, dim_train)

for idx in lista_indici:
    if idx not in index_train:
        index_test.append(idx)

random.shuffle(index_test)

paz_train = []
paz_test = []
for j in index_train:
    paz = dati_nome_chiave[j]
    paz_train.append(paz)

for j in index_test:
    paz = dati_nome_chiave[j]
    paz_test.append(paz)


xtrain = []
xtest = []


for j in index_train:
    paz = dati_skeleton_body[j]
    xtrain.append(paz)


for j in index_test:
    paz = dati_skeleton_body[j]
    xtest.append(paz)


dizionario_train = {}
dizionario_test = {}

for i in range(len(paz_train)):
    dizionario_train[paz_train[i]] = xtrain[i]

for i in range(len(paz_test)):
    dizionario_test[paz_test[i]] = xtest[i]


with open('train_ncv_ncs_dl', 'wb') as file:
    pickle.dump(dizionario_train, file)

with open('test_ncv_ncs_dl', 'wb') as file:
    pickle.dump(dizionario_test, file)