import pandas as pd
import numpy as np
import random
import pickle


Tabella = pd.read_csv('tabella_features.csv')

indici_vista_frontale2 = []
nomi_vista_frontale2 = []

for index, row in Tabella.iterrows():
    if "C002" in row.iloc[0]:
        if "R002" in row.iloc[0]:
            indici_vista_frontale2.append(index)
            nomi_vista_frontale2.append(row.iloc[0])

Tabella = Tabella.iloc[indici_vista_frontale2,:]
Tabella.reset_index(drop=True, inplace=True)
Tabella.index = Tabella.index + 1

# adesso faccio la suddivisione cross subject

dim_train_cv_cs = []
dim_test_cv_cs = []



def estrai_paz(stringa):
    inizio = stringa.index('P')
    fine = inizio + 4

    return stringa[inizio:fine]

lista_paz = []
for index,row in Tabella.iterrows():
    paz = estrai_paz(row.iloc[0])
    lista_paz.append(paz)

pazienti = sorted(set(lista_paz))

for i in range(100):

    paz_train = random.sample(pazienti, k = 24)
    paz_test = []

    for paz in pazienti:
        if paz not in paz_train:
            paz_test.append(paz)


# faccio una prima suddivisione del train e test in base ai paz
    paz_train = sorted(paz_train)
    paz_test = sorted(paz_test)
    indici_train_cross_sub = []
    indici_test_cross_sub = []

    for paz in paz_train:
        for index,row in Tabella.iterrows():
            if paz in row.iloc[0]:
                indici_train_cross_sub.append(index)

    for paz in paz_test:
        for index,row in Tabella.iterrows():
            if paz in row.iloc[0]:
                indici_test_cross_sub.append(index)

    Tabella_train_cross_subject = Tabella.loc[indici_train_cross_sub,:]
    Tabella_test_cross_subject = Tabella.loc[indici_test_cross_sub,:]

# adesso faccio la suddivisione cross setting


    def estrai_sett(stringa):
        inizio = stringa.index("S")
        fine = inizio + 4

        return stringa[inizio:fine]

    lista_set = []
    for index,row in Tabella.iterrows():
        setting = estrai_sett(row.iloc[0])
        lista_set.append(setting)

    setting = sorted(set(lista_set))
    set_train = random.sample(setting, k = 10)
    set_test = []

    for sett in setting:
        if sett not in set_train:
            set_test.append(sett)

    set_train = sorted(set_train)
    set_test = sorted(set_test)

    indici_train_cross_sub_set = []
    for sett in set_train:
        for index,row in Tabella_train_cross_subject.iterrows():
            if sett in row.iloc[0]:
                indici_train_cross_sub_set.append(index)

    indici_test_cross_sub_set = []
    for sett in set_test:
        for index,row in Tabella_test_cross_subject.iterrows():
            if sett in row.iloc[0]:
                indici_test_cross_sub_set.append(index)


    Tabella_train_cross_subject_set = Tabella_train_cross_subject.loc[indici_train_cross_sub_set,:]
    Tabella_test_cross_subject_set = Tabella_test_cross_subject.loc[indici_test_cross_sub_set,:]

    dim_train = len(indici_train_cross_sub_set)
    dim_test = len(indici_test_cross_sub_set)

    dim_train_cv_cs.append(dim_train)
    dim_test_cv_cs.append(dim_test)

    print(i)

#Tabella_train_cross_subject_set.to_csv('Tabella_train_cross_subject_set.csv')
#Tabella_test_cross_subject_set.to_csv('Tabella_test_cross_subject_set.csv')

with open('dim_train_cv_cs.pkl', 'wb') as f:
    pickle.dump(dim_train_cv_cs, f)


with open('dim_test_cv_cs.pkl', 'wb') as f:
    pickle.dump(dim_test_cv_cs, f)




