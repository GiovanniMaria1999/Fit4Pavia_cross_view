import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
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

def estrai_sett(s):
    inizio = s.index("S")
    fine = inizio+4
    return s[inizio:fine]

setting = []
for data in dati_nome_file:
    substring = estrai_sett(data) # iloc vuole l'indice intero della posizione
    setting.append(substring)

lista_set = sorted(set(setting))


conf_matrix = np.zeros((2,2),dtype = int)

dim_train_sim = []
dim_test_sim = []
etichetta_is_tot = []
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


    set_train = sorted(random.sample(lista_set, 12))  # restituisce casualmente gli elementi della lista senza ripetizioni
    set_test = []
    for set in lista_set:
        if set not in set_train:
            set_test.append(set)


    idx_train = []
    idx_test = []

    for set in set_train:
        for index, nome in enumerate(dati_nome_chiave):  # enumerate ritorna l'indice e l'elemento
            if set in nome:
                idx_train.append(index)

    for set in set_test:
        for index, nome in enumerate(dati_nome_chiave):
            if set in nome:
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
    etichetta_is_tot_train.append(paz_train)
    etichetta_is_tot_test.append(paz_test)
    lista_nomi_sim.append(lista_nomi)



    lista_len = []
    for i in range(len(dati_skeleton_body)):
        lungh = len(dati_skeleton_body[i])
        lista_len.append(lungh)

    max_lungh = max(lista_len)
    tabella_padding = []

    for vet in dati_skeleton_body:
        padding_size = max_lungh - len(vet)
        vettore = torch.tensor(vet)
        padding_vet = torch.nn.functional.pad(vettore, (0, 0, 0, padding_size))
        tabella_padding.append(padding_vet)

    tabella = torch.stack(tabella_padding)
    tabella = tabella.to(torch.float32)

    # faccio la trasposizione
    tabella = tabella.permute(0, 2, 1)

    xtrain = []
    xtest = []

    xtrain = tabella[idx_train]
    xtest = tabella[idx_test]

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

    ytrain = torch.tensor(ytrain)
    ytrain = ytrain.to(torch.float32)
    ytrain = ytrain.unsqueeze(1)
    ytest = torch.tensor(ytest)


############### costruisco la rete


    in_channels = 75
    out_channels = 64
    num_classes = 1
    hidden_size = 100


    class CONV1DLSTM(nn.Module):
        def __init__(self, in_channels, out_channels, hidden_size, num_classes):
            super(CONV1DLSTM, self).__init__()

            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.pool1 = nn.AvgPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=128, kernel_size=3, padding=1)
            self.pool2 = nn.AvgPool1d(kernel_size=2)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.pool3 = nn.AvgPool1d(kernel_size=2)
            self.relu = nn.ReLU()

            self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.3,bidirectional=False)
            self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

        def forward(self, x):
            x = self.relu(self.pool1(self.conv1(x)))
            x = self.relu(self.pool2(self.conv2(x)))
            x = self.relu(self.pool3(self.conv3(x)))

            x = x.permute(0, 2, 1)

            lstm_out, (h_n, c_n) = self.lstm(x)
            out = lstm_out[:, -1, :]
            out = self.fc(out)

            return out


    model = CONV1DLSTM(in_channels, out_channels, hidden_size, num_classes)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    iperparametri = {
        "learning_rate": 0.001,
        "epoche": 100,
        "hidden_size": 100,
        "in_channels": 75,
        "out_channels": 64,
        "num_classes": 1,
        "loss_function": "BCEWithLogitsLoss",
        "optimizer": "Adam"
    }

    epoche = 100

    for epoca in range(epoche):
        model.train()

        optimizer.zero_grad()

        output = model(xtrain)
        loss = criterion(output, ytrain)
        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
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
    print(f"accuratezza", accuratezza_train, f"f1_score", f1_train)

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
    conf_matrix += confusion_matrix(ytest, predizioni_test)

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


conf_matrix = np.round(conf_matrix/100)


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

with open(os.path.join(cartella,'accuratezza_conv1dlstm_cv_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_test,file)

with open(os.path.join(cartella, 'accuratezza_conv1dlstm_cv_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_train, file)

with open(os.path.join(cartella,'f1_conv1dlstm_cv_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_test,file)

with open(os.path.join(cartella,'f1_conv1dlstm_cv_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_train, file)

with open(os.path.join(cartella,'confmatrix_conv1dlstm_cv.pkl'), 'wb') as file:
    pickle.dump(conf_matrix, file)


with open(os.path.join(cartella,'conv1dlstm_simul_id_cv.pkl'), 'wb') as file:
    pickle.dump(array_simul_id,file)

with open(os.path.join(cartella,'conv1dlstm_array_dim_train_cv.pkl'), 'wb') as file:
    pickle.dump(array_dim_train_id,file)

with open(os.path.join(cartella,'conv1dlstm_array_dim_test_cv.pkl'), 'wb') as file:
    pickle.dump(array_dim_test_id,file)


with open(os.path.join(cartella,'conv1dlstm_etichetta_is_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot, file)

with open(os.path.join(cartella,'conv1dlstm_etichetta_is_train_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_train, file)

with open(os.path.join(cartella,'conv1dlstm_etichetta_is_test_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_test, file)


with open(os.path.join(cartella,'conv1dlstm_lista_nomi_simulazione_cv.pkl'), 'wb') as file:
    pickle.dump(lista_nomi_sim,file)

with open(os.path.join(cartella,'conv1dlstm_is_pred_cc_train_cv.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_train_tot,file)

with open(os.path.join(cartella,'conv1dlstm_is_pred_cc_test_cv.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_test_tot,file)

with open(os.path.join(cartella,'conv1dlstm_item_loss_test_cv.pkl'), 'wb') as file:
    pickle.dump(item_loss_test_tot,file)

with open(os.path.join(cartella,'conv1dlstm_item_loss_train_cv.pkl'), 'wb') as file:
    pickle.dump(item_loss_train_tot,file)

with open(os.path.join(cartella,'conv1dlstm_prob_conf_train_cv.pkl'), 'wb') as file:
    pickle.dump(lista_prob_pred_train,file)

with open(os.path.join(cartella,'conv1dlstm_prob_conf_test_cv.pkl'), 'wb') as file:
    pickle.dump(lista_prob_pred_test,file)

with open(os.path.join(cartella,'conv1dlstm_ytest_cv.pkl'), 'wb') as file:
    pickle.dump(ytest_tot,file)

with open(os.path.join(cartella,'conv1dlstm_ytrain_cv.pkl'), 'wb') as file:
    pickle.dump(ytrain_tot,file)


with open(os.path.join(cartella,"iperparametri_conv1dlstm_cv.pkl"),"wb") as file:
    pickle.dump(iperparametri, file)


with open(os.path.join(cartella,'conv1dlstm_pred_train_cv.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train,file)

with open(os.path.join(cartella,'conv1dlstm_pred_test_cv.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_test,file)