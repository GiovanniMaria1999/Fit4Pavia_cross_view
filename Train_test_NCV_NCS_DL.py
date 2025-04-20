import pickle
import numpy as np
import random


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

dati_nome_file = list(dizionario.keys())
dati_skeleton = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste

nome_vista_fron2 = []
indici_vista_fron2 = []
for index, nome in enumerate(dati_nome_file):
    if "C002" in nome:
        if "R002" in nome:
            nome_vista_fron2.append(nome)
            indici_vista_fron2.append(index)


dati_skeleton_body = []
for i in indici_vista_fron2:
    dati = dati_skeleton[i]
    dati_skeleton_body.append(dati)

dati_nome_chiave = []
for i in indici_vista_fron2:
    dati = nome_chiave[i]
    dati_nome_chiave.append(dati)


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