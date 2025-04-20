import pickle
import random

with open('dizionario_mc_vf2.pkl', 'rb') as file:
    dizionario = pickle.load(file)


dati_nome_chiave = list(dizionario.keys())
dati_skeleton_body = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste


def estrai_paz(stringa):
    inizio = stringa.index("P")
    fine = inizio + 4

    return stringa[inizio:fine]


lista_paz = []
for nome in dati_nome_chiave:
    paziente = estrai_paz(nome)
    lista_paz.append(paziente)

pazienti = sorted(set(lista_paz))


paz_train = random.sample(pazienti, k=75)
paz_train = sorted(paz_train)
paz_test = []

for paz in pazienti:
    if paz not in paz_train:
        paz_test.append(paz)

indici_cross_subject_train = []
nome_cross_subject_train = []
for paziente in paz_train:
    for index, nome in enumerate(dati_nome_chiave):
        if paziente in nome:
            indici_cross_subject_train.append(index)
            nome_cross_subject_train.append(nome)

indici_cross_subject_test = []
nome_cross_subject_test = []
for paziente in paz_test:
    for index, nome in enumerate(dati_nome_chiave):
        if paziente in nome:
            indici_cross_subject_test.append(index)
            nome_cross_subject_test.append(nome)


dati_skel_cross_subject_train = []
for index in indici_cross_subject_train:
    dati = dati_skeleton_body[index]
    dati_skel_cross_subject_train.append(dati)

dati_skel_cross_subject_test = []
for index in indici_cross_subject_test:
    dati = dati_skeleton_body[index]
    dati_skel_cross_subject_test.append(dati)

xtrain = dati_skel_cross_subject_train
xtest = dati_skel_cross_subject_test

dizionario_train = {}
dizionario_test = {}

for i in range(len(paz_train)):
    dizionario_train[paz_train[i]] = xtrain[i]

for i in range(len(paz_test)):
    dizionario_test[paz_test[i]] = xtest[i]

with open('train_ncv_cs_dl', 'wb') as file:
    pickle.dump(dizionario_train, file)

with open('test_ncv_cs_dl', 'wb') as file:
    pickle.dump(dizionario_test, file)