import pandas as pd
import random
import pickle

Tabella = pd.read_csv('tabella_features.csv')

# parto col trovare i dati solo nella vista frontale camera 2 (ripetizione 2)

indici_vista_frontale2 = []
for index,row in Tabella.iterrows():
    if "C002" in row.iloc[0]:
        if "R002" in row.iloc[0]:
            indici_vista_frontale2.append(index)

Tabella = Tabella.iloc[indici_vista_frontale2,:]
Tabella.reset_index(drop=True,inplace = True) # resetto gli indici della tabella da 1 a n
Tabella.index = Tabella.index + 1

def estrai_sett(s):
    inizio = s.index("S")
    fine = inizio+4
    return s[inizio:fine]

setting = []
for index,row in Tabella.iterrows():
    substring = estrai_sett(row.iloc[0]) # iloc vuole l'indice intero della posizione
    setting.append(substring)

lista_set = sorted(set(setting))

dim_train_cv = []
dim_test_cv = []

for i in range(100):
    set_train = sorted(random.sample(lista_set,12)) # restituisce casualmente gli elementi della lista senza ripetizioni
    set_test = []
    for set in lista_set:
        if set not in set_train:
            set_test.append(set)

#print(Tabella)

    idx_train = []
    idx_test = []

    for set in set_train:
        for index,row in Tabella.iterrows():
            if set in row.iloc[0]:
                idx_train.append(index)


    for set in set_test:
        for index,row in Tabella.iterrows():
            if set in row.iloc[0]:
                idx_test.append(index)

# ora che ho gli indici di train e test, ricavo le tabelle di train e test

    tabella_train = Tabella.loc[idx_train,:] # loc usa l'indice reale, iloc parte da 0
    tabella_test = Tabella.loc[idx_test,:]

    dim_train = len(idx_train)
    dim_test = len(idx_test)

    dim_train_cv.append(dim_train)
    dim_test_cv.append(dim_test)

    print(i)

#tabella_train.to_csv("tabella_train_set.csv")
#tabella_test.to_csv("tabella_test_set.csv")

with open('dim_train_cv.pkl', 'wb') as f:
    pickle.dump(dim_train_cv, f)


with open('dim_test_cv.pkl', 'wb') as f:
    pickle.dump(dim_test_cv, f)
