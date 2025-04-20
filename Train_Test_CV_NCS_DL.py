import pickle
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


def estrai_sett(s):
    inizio = s.index("S")
    fine = inizio+4
    return s[inizio:fine]

setting = []
for data in dati_nome_chiave:
    substring = estrai_sett(data) # iloc vuole l'indice intero della posizione
    setting.append(substring)

lista_set = sorted(set(setting))


set_train = random.sample(lista_set,12) # restituisce casualmente gli elementi della lista senza ripetizioni
set_test = []
for sett in lista_set:
    if sett not in set_train:
        set_test.append(sett)

idx_train = []
idx_test = []

for sett in set_train:
    for index, nome in enumerate(dati_nome_chiave): # enumerate ritorna l'indice e l'elemento
        if sett in nome:
            idx_train.append(index)

for sett in set_test:
    for index,nome in enumerate(dati_nome_chiave):
        if sett in nome:
            idx_test.append(index)

xtrain = []
xtest = []

for j in idx_train:
    dati = dati_skeleton_body[j]
    xtrain.append(dati)

for j in idx_test:
    dati = dati_skeleton_body[j]
    xtest.append(dati)

nome_chiave_train = []
nome_chiave_test = []

for j in idx_train:
    nome = dati_nome_chiave[j]
    nome_chiave_train.append(nome)

for j in idx_test:
    nome = dati_nome_chiave[j]
    nome_chiave_test.append(nome)

dizionario_train = {}
dizionario_test = {}

for i in range(len(nome_chiave_train)):
    dizionario_train[nome_chiave_train[i]] = xtrain[i]

for i in range(len(nome_chiave_test)):
    dizionario_test[nome_chiave_test[i]] = xtest[i]