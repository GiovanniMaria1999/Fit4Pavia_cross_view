import pandas as pd
import random
import pickle

Tabella = pd.read_csv('tabella_features.csv')

# Suddivisione in train e test nel caso non cross-subject per i modelli ML

indici_vista_frontale2 = []
for index,row in Tabella.iterrows():
    if "C002" in row.iloc[0]:
        if "R002" in row.iloc[0]:
            indici_vista_frontale2.append(index)

Tabella = Tabella.iloc[indici_vista_frontale2,:]
Tabella.reset_index(drop=True,inplace = True) # resetto gli indici della tabella da 1 a n
Tabella.index = Tabella.index + 1

lista_index = []
for index,row in Tabella.iterrows():
    lista_index.append(index)

dim_train_ncs = []
dim_test_ncs = []

for i in range(100):

    dim_train = round(70*len(lista_index)/100)
    index_train = sorted(random.sample(lista_index, dim_train))

    index_test = []
    for idx in lista_index:
        if idx not in index_train:
            index_test.append(idx)

    dim_test = len(index_test)


# adesso devo creare x_train, x_test, y_train e y_test, y_train deve essere un vettore di 0,1, idem y_test
# 0 per sit down, 1 stand up

    tabella_train = Tabella.loc[index_train,:]
    tabella_test = Tabella.loc[index_test,:]

    dim_train_ncs.append(dim_train)
    dim_test_ncs.append(dim_test)

    print(i)

#tabella_train.to_csv("tabella_train.csv")
#tabella_test.to_csv("tabella_test.csv")

with open('dim_train_ncs.pkl', 'wb') as f:
    pickle.dump(dim_train_ncs, f)


with open('dim_test_ncs.pkl', 'wb') as f:
    pickle.dump(dim_test_ncs, f)









