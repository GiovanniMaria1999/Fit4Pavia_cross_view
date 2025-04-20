import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os


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
dati_skeleton = list(dizionario.values())


indici_vista_fron2 = []
for index, nome in enumerate(dati_nome_file):
    if "C002" in nome:
        if "R002" in nome:
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
for data in dati_nome_file:
    substring = estrai_sett(data) # iloc vuole l'indice intero della posizione
    setting.append(substring)

lista_set = sorted(set(setting))

conf_matrix_lstm = np.zeros((2,2),dtype = int)

dim_train_sim = []
dim_test_sim = []
etichetta_is_tot = []
etichetta_is_cv_tot = []
etichetta_is_tot_train = []
etichetta_is_tot_test = []
lista_nomi_sim = []
ytest_tot = []
ytrain_tot = []

lista_acc_train = []
lista_acc_test = []

lista_f1_train = []
lista_f1_test = []

lista_predizioni_train = []
lista_predizioni_test = []

lista_prob_pred_train = []
lista_prob_pred_test = []

is_pred_cc_test_tot = []
item_loss_test_tot = []

is_pred_cc_train_tot = []
item_loss_train_tot = []

for i in range(100):

    lista_nomi = []
    for nome in dati_nome_chiave:
        lista_nomi.append(nome)

    etichetta_tot = []
    for j in range(len(indici_vista_fron2)):
        etich = "FALSE"
        etichetta_tot.append(etich)

    etichetta_cv_tot = []
    for j in range(len(indici_vista_fron2)):
        etich = "TRUE"
        etichetta_cv_tot.append(etich)

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

    paz_train = []
    paz_test = []
    for j in idx_train:
        paz = dati_nome_chiave[j]
        paz_train.append(paz)

    for j in idx_test:
        paz = dati_nome_chiave[j]
        paz_test.append(paz)

    dim_train = len(idx_train)
    dim_test = len(idx_test)

    dim_train_sim.append(dim_train)
    dim_test_sim.append(dim_test)

    etichetta_is_tot.append(etichetta_tot)
    etichetta_is_cv_tot.append(etichetta_cv_tot)
    etichetta_is_tot_train.append(paz_train)
    etichetta_is_tot_test.append(paz_test)
    lista_nomi_sim.append(lista_nomi)


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

    ytrain = []
    ytest = []

    for nome in nome_chiave_train:
        if "A008" in nome:
            y = 0

        elif "A009" in nome:
            y = 1

        ytrain.append(y)

    for nome in nome_chiave_test:
        if "A008" in nome:
            y = 0

        elif "A009" in nome:
            y = 1

        ytest.append(y)

    ytest_tot.append(ytest)
    ytrain_tot.append(ytrain)


# effettuo lo zero padding in modo che tutte le sequenze abbiano la stessa lunghezza

    lista_len = []
    for j in range(len(idx_train)):
        lungh = len(xtrain[j])
        lista_len.append(lungh)


    max_lungh = max(lista_len)

    xtrain_padding = []

    for vet in xtrain:
        padding_size = max_lungh - len(vet)
        vettore = torch.tensor(vet)
        padding_vet = torch.nn.functional.pad(vettore, (0,0,0, padding_size)) # (0,0,0) serve per aggiungere righe di 0 in fondo alla matrice
    #(0,0) significa che non voglio aggiungere 0 sulle righe orizzontali, il terzo 0 non voglio 0 in alto
        xtrain_padding.append(padding_vet)

    xtrain = torch.stack(xtrain_padding) # concateno la sequenza di tensori
    xtrain = xtrain.to(torch.float32) # converto il tensore in float32

    for j in range(len(idx_test)):
        lungh = len(xtest[j])
        lista_len.append(lungh)

    max_lungh = max(lista_len)

    xtest_padding = []

    for vet in xtest:
        padding_size = max_lungh - len(vet)
        vettore = torch.tensor(vet)
        padding_vet = torch.nn.functional.pad(vettore, (0, 0, 0, padding_size))
        xtest_padding.append(padding_vet)

    xtest = torch.stack(xtest_padding)
    xtest = xtest.to(torch.float32)

    ytrain = torch.tensor(ytrain)
    ytrain = ytrain.to(torch.float32)
    ytrain = ytrain.unsqueeze(1)


# creazione rete LSTM

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
            super(LSTM, self).__init__()

            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,dropout=dropout, bidirectional=False)  # hidden size numero di neuroni
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            out = lstm_out[:, -1, :]  # prendo l'ultimo passo temporale (.detach)
            out = self.fc(out)

            return out


    input_size = 75
    hidden_size = 100
    num_layers = 3
    num_classes = 1
    dropout = 0.3

    model = LSTM(input_size, hidden_size, num_layers, dropout, num_classes)

    iperparametri = {
        "learning_rate": 0.001,
        "epoche": 100,
        "input_size": 75,
        "num_layers": 3,
        "hidden_size": 100,
        "num_classes": 1,
        "dropout": 0.3,
        "loss_function": "BCEWithLogitsLoss",
        "optimizer": "Adam"
    }

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoche = 100

    for epoch in range(epoche):
        model.train()

        # azzeramento dei gradienti
        optimizer.zero_grad()

        out = model(xtrain)
        loss = criterion(out, ytrain)

        # backpropagation e ottimizzazione

        loss.backward()  # calcola i gradienti della perdita rispetto ai parametri del modello
        optimizer.step()  # aggiorno i pesi del modello e minimizza la perdita

        # calcolo le predizioni


    model.eval()  # imposto il modello in modalità valutazione
    with torch.no_grad():  # disabilito il calcolo del gradiente

        output = model(xtrain)
        predizioni_prob_train = torch.sigmoid(output)
        predicted_classes = (predizioni_prob_train > 0.5).float()

    predizioni_train = predicted_classes.to(torch.long)  # così ottengo un intero
    predizioni_train = np.array(predizioni_train)
    predizioni_train = predizioni_train.flatten()  # trasforma l'array nidificato in un array monodimensionale

    lista_predizioni_train.append(predizioni_train)
    lista_prob_pred_train.append(predizioni_prob_train)

    ytrain = ytrain.squeeze(1)
    ytrain = np.array(ytrain)
    accuratezza_train = accuracy_score(ytrain, predizioni_train)
    f1_train = f1_score(ytrain, predizioni_train)
    #print(f"accuratezza", accuratezza_train, f"f1_score", f1_train)

    lista_acc_train.append(accuratezza_train)
    lista_f1_train.append(f1_train)

    is_pred_cc_train = []
    item_loss_train = []

    for j in range(len(ytrain)):
        if ytrain[j] == predizioni_train[j]:
            is_pred = 1
            loss = 0
        elif ytrain[j] != predizioni_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_train.append(is_pred)
        item_loss_train.append(loss)

    is_pred_cc_train_tot.append(is_pred_cc_train)
    item_loss_train_tot.append(item_loss_train)

    model.eval()
    with torch.no_grad():
        output = model(xtest)
        predizioni_prob_test = torch.sigmoid(output)
        predicted_classes = (predizioni_prob_test > 0.5).float()

    predizioni_test = predicted_classes.to(torch.long)  # così ottengo un intero
    predizioni_test = np.array(predizioni_test)
    predizioni_test = predizioni_test.flatten()  # trasforma l'array nidificato in un array monodimensionale

    lista_predizioni_test.append(predizioni_test)
    lista_prob_pred_test.append(predizioni_prob_test)

    ytest = np.array(ytest)
    accuratezza_test = accuracy_score(ytest, predizioni_test)
    f1_test = f1_score(ytest, predizioni_test)
    print(f"accuratezza", accuratezza_test, f"f1_score", f1_test)
    lista_acc_test.append(accuratezza_test)
    lista_f1_test.append(f1_test)
    conf_matrix_lstm += confusion_matrix(ytest, predizioni_test)

    is_pred_cc_test = []
    item_loss_test = []

    for j in range(len(ytest)):
        if ytest[j] == predizioni_test[j]:
            is_pred = 1
            loss = 0
        elif ytest[j] != predizioni_test[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_test.append(is_pred)
        item_loss_test.append(loss)

    is_pred_cc_test_tot.append(is_pred_cc_test)
    item_loss_test_tot.append(item_loss_test)

    print(i)


conf_matrix_lstm = np.round(conf_matrix_lstm/100)


vettore_tot = []
for i in range(100):
    vettore = np.full(314, i)
    vettore_tot.append(vettore)

array_simul_id = np.array(vettore_tot)
array_simul_id = array_simul_id.reshape(-1, 1)  # ottengo un vettore colonna con 314*100 righe


vettore_tot = []
for dim in dim_train_sim:
    vettore = np.full(314, dim)
    vettore_tot.append(vettore)

array_dim_train_id = np.array(vettore_tot)
array_dim_train_id = array_dim_train_id.reshape(-1, 1)

vettore_tot = []
for dim in dim_test_sim:
    vettore = np.full(314, dim)
    vettore_tot.append(vettore)

array_dim_test_id = np.array(vettore_tot)
array_dim_test_id = array_dim_test_id.reshape(-1, 1)

cartella = "modelli"

with open(os.path.join(cartella,'accuratezza_bilstm_cv_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_test,file)

with open(os.path.join(cartella, 'accuratezza_bilstm_cv_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_train, file)

with open(os.path.join(cartella,'f1_bilstm_cv_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_test,file)

with open(os.path.join(cartella,'f1_bilstm_cv_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_train, file)

with open(os.path.join(cartella,'confmatrix_bilstm_cv.pkl'), 'wb') as file:
    pickle.dump(conf_matrix_lstm, file)

with open(os.path.join(cartella,'bilstm_simul_id_cv.pkl'), 'wb') as file:
    pickle.dump(array_simul_id,file)

with open(os.path.join(cartella,'bilstm_array_dim_train_cv.pkl'), 'wb') as file:
    pickle.dump(array_dim_train_id,file)

with open(os.path.join(cartella,'bilstm_array_dim_test_cv.pkl'), 'wb') as file:
    pickle.dump(array_dim_test_id,file)


with open(os.path.join(cartella,'bilstm_etichetta_is_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot, file)

with open(os.path.join(cartella,'bilstm_etichetta_is_crossview_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_cv_tot, file)

with open(os.path.join(cartella,'bilstm_etichetta_is_train_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_train, file)

with open(os.path.join(cartella,'bilstm_etichetta_is_test_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_test, file)


with open(os.path.join(cartella,'bilstm_lista_nomi_simulazione_cv.pkl'), 'wb') as file:
    pickle.dump(lista_nomi_sim,file)

with open(os.path.join(cartella,'bilstm_is_pred_cc_train_cv.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_train_tot,file)

with open(os.path.join(cartella,'bilstm_is_pred_cc_test_cv.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_test_tot,file)

with open(os.path.join(cartella,'bilstm_item_loss_test_cv.pkl'), 'wb') as file:
    pickle.dump(item_loss_test_tot,file)

with open(os.path.join(cartella,'bilstm_item_loss_train_cv.pkl'), 'wb') as file:
    pickle.dump(item_loss_train_tot,file)

with open(os.path.join(cartella,'bilstm_prob_conf_train_cv.pkl'), 'wb') as file:
    pickle.dump(lista_prob_pred_train,file)

with open(os.path.join(cartella,'bilstm_prob_conf_test_cv.pkl'), 'wb') as file:
    pickle.dump(lista_prob_pred_test,file)

with open(os.path.join(cartella,'bilstm_ytest_cv.pkl'), 'wb') as file:
    pickle.dump(ytest_tot,file)

with open(os.path.join(cartella,'bilstm_ytrain_cv.pkl'), 'wb') as file:
    pickle.dump(ytrain_tot,file)

with open("iperparametri_bilstm_cv.pkl","wb") as file:
    pickle.dump(iperparametri, file)

with open(os.path.join(cartella,'bilstm_pred_train_cv.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train,file)

with open(os.path.join(cartella,'bilstm_pred_test_cv.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_test,file)

