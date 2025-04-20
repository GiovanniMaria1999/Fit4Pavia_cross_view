import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import os

with open('dizionario.pkl', 'rb') as f:
    dizionario = pickle.load(f)

dati_skeleton_body = list(dizionario.values())
nome_chiave = list(dizionario.keys())

lista_indici_vf2 = []

for index, nome in enumerate(nome_chiave):
    if "C002" in nome:
        if "R002" in nome:
            lista_indici_vf2.append(index)


dati_skeleton_vf2 = []

for i in lista_indici_vf2:
    dati = dati_skeleton_body[i]
    dati_skeleton_vf2.append(dati)

dati_nome_chiave = []
for i in lista_indici_vf2:
    nomi = nome_chiave[i]
    dati_nome_chiave.append(nomi)

def estrai_set(stringa):
    inizio = stringa.index("S")
    fine = inizio + 4

    return stringa[inizio:fine]

lista_set = []
for nome in dati_nome_chiave:
    setting = estrai_set(nome)
    lista_set.append(setting)

setting = sorted(set(lista_set))


def estrai_paz(stringa):
    inizio = stringa.index("P")
    fine = inizio + 4

    return stringa[inizio:fine]


lista_paz = []
for nome in dati_nome_chiave:
    paziente = estrai_paz(nome)
    lista_paz.append(paziente)

pazienti = sorted(set(lista_paz))

conf_matrix = np.zeros((2,2),dtype = int)


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

    # devo fare il padding

    lista_nomi = []
    for nome in dati_nome_chiave:
        lista_nomi.append(nome)

    etichetta_tot = []
    for j in range(len(lista_indici_vf2)):
        etich = "TRUE"
        etichetta_tot.append(etich)

    etichetta_cv_tot = []
    for j in range(len(lista_indici_vf2)):
        etich = "TRUE"
        etichetta_cv_tot.append(etich)

    lista_len = []
    for dati in dati_skeleton_vf2:
        lunghezza = len(dati)
        lista_len.append(lunghezza)

    lungh_max = max(lista_len)

    tabella_padding = []
    for vett in dati_skeleton_vf2:
        padding_size = lungh_max - len(vett)
        vettore = torch.tensor(vett)
        padding_vet = nn.functional.pad(vettore, (0,0,0,padding_size))
        tabella_padding.append(padding_vet)

    tabella = torch.stack(tabella_padding) # concateno la sequenza di tensori
    tabella = tabella.to(torch.float32)

    tabella = tabella.permute(0,2,1)

    paz_train = random.sample(pazienti, k=24)
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

    train_cross_subject = tabella[indici_cross_subject_train]
    test_cross_subject = tabella[indici_cross_subject_test]

    set_train = random.sample(setting, k=10)
    set_train = sorted(set(set_train))

    set_test = []
    for sett in setting:
        if sett not in set_train:
            set_test.append(sett)

    indici_cross_subject_cross_view_train = []
    nome_cross_subject_cross_view_train = []
    for sett in set_train:
        for index, nome in enumerate(nome_cross_subject_train):
            if sett in nome:
                indici_cross_subject_cross_view_train.append(index)
                nome_cross_subject_cross_view_train.append(nome)

    indici_cross_subject_cross_view_test = []
    nome_cross_subject_cross_view_test = []
    for sett in set_test:
        for index, nome in enumerate(nome_cross_subject_test):
            if sett in nome:
                indici_cross_subject_cross_view_test.append(index)
                nome_cross_subject_cross_view_test.append(nome)


    train_cross_subject_cross_view = train_cross_subject[indici_cross_subject_cross_view_train]
    test_cross_subject_cross_view = test_cross_subject[indici_cross_subject_cross_view_test]

    xtrain = train_cross_subject_cross_view
    xtest = test_cross_subject_cross_view

    paz_train = []
    paz_test = []
    for j in indici_cross_subject_cross_view_train:
        paz = nome_cross_subject_train[j]
        paz_train.append(paz)

    for j in indici_cross_subject_cross_view_test:
        paz = nome_cross_subject_test[j]
        paz_test.append(paz)

    dim_train = len(indici_cross_subject_cross_view_train)
    dim_test = len(indici_cross_subject_cross_view_test)

    dim_train_sim.append(dim_train)
    dim_test_sim.append(dim_test)

    etichetta_is_tot.append(etichetta_tot)
    etichetta_is_cv_tot.append(etichetta_cv_tot)
    etichetta_is_tot_train.append(paz_train)
    etichetta_is_tot_test.append(paz_test)
    lista_nomi_sim.append(lista_nomi)

    ytrain = []
    ytest = []

    for nome in nome_cross_subject_cross_view_train:
        if "A008" in nome:
            y = 0
        elif "A009" in nome:
            y = 1

        ytrain.append(y)

    for nome in nome_cross_subject_cross_view_test:
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


    xtrain = xtrain.unsqueeze(1)
    xtest = xtest.unsqueeze(1)

    in_channels = 1
    out_channels = 64
    num_classes = 1
    hidden_size = 100


    class CONV2DLSTM(nn.Module):
        def __init__(self, in_channels, out_channels, hidden_size, num_classes):
            super(CONV2DLSTM, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = (3, 3), stride = 2, padding=1)
            self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=128, kernel_size=(3,3), stride = 2,padding=1)
            self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride = 2, padding=1)
            self.pool3 = nn.AvgPool2d(kernel_size=(2,2))
            self.relu = nn.ReLU()

            self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.3,bidirectional=False)
            self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

        def forward(self, x):
            x = self.relu(self.pool1(self.conv1(x)))
            x = self.relu(self.pool2(self.conv2(x)))
            x = self.relu(self.pool3(self.conv3(x)))

            x = torch.flatten(x, start_dim=2)
            x = x.permute(0, 2, 1)

            lstm_out, (h_n, c_n) = self.lstm(x)
            out = lstm_out[:, -1, :]
            out = self.fc(out)

            return out


    model = CONV2DLSTM(in_channels, out_channels, hidden_size, num_classes)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    iperparametri = {
        "learning_rate": 0.001,
        "epoche": 100,
        "hidden_size": 100,
        "in_channels": 1,
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


conf_matrix = np.round(conf_matrix / 100)



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

with open(os.path.join(cartella,'accuratezza_conv2dlstm_cs_cv_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_test,file)

with open(os.path.join(cartella, 'accuratezza_conv2dlstm_cs_cv_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_train, file)

with open(os.path.join(cartella,'f1_conv2dlstm_cs_cv_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_test,file)

with open(os.path.join(cartella,'f1_conv2dlstm_cs_cv_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_train, file)

with open(os.path.join(cartella,'confmatrix_conv2dlstm_cs_cv.pkl'), 'wb') as file:
    pickle.dump(conf_matrix , file)


with open(os.path.join(cartella,'conv2dlstm_simul_id_cs_cv.pkl'), 'wb') as file:
    pickle.dump(array_simul_id,file)

with open(os.path.join(cartella,'conv2dlstm_array_dim_train_cs_cv.pkl'), 'wb') as file:
    pickle.dump(array_dim_train_id,file)

with open(os.path.join(cartella,'conv2dlstm_array_dim_test_cs_cv.pkl'), 'wb') as file:
    pickle.dump(array_dim_test_id,file)


with open(os.path.join(cartella,'conv2dlstm_etichetta_is_cs_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot, file)

with open(os.path.join(cartella,'conv2dlstm_etichetta_is_crossview_cs_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_cv_tot, file)

with open(os.path.join(cartella,'conv2dlstm_etichetta_is_train_cs_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_train, file)

with open(os.path.join(cartella,'conv2dlstm_etichetta_is_test_cs_cv.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_test, file)


with open(os.path.join(cartella,'conv2dlstm_lista_nomi_simulazione_cs_cv.pkl'), 'wb') as file:
    pickle.dump(lista_nomi_sim,file)

with open(os.path.join(cartella,'conv2dlstm_is_pred_cc_train_cs_cv.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_train_tot,file)

with open(os.path.join(cartella,'conv2dlstm_is_pred_cc_test_cs_cv.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_test_tot,file)

with open(os.path.join(cartella,'conv2dlstm_item_loss_test_cs_cv.pkl'), 'wb') as file:
    pickle.dump(item_loss_test_tot,file)

with open(os.path.join(cartella,'conv2dlstm_item_loss_train_cs_cv.pkl'), 'wb') as file:
    pickle.dump(item_loss_train_tot,file)

with open(os.path.join(cartella,'conv2dlstm_prob_conf_train_cs_cv.pkl'), 'wb') as file:
    pickle.dump(lista_prob_pred_train,file)

with open(os.path.join(cartella,'conv2dlstm_prob_conf_test_cs_cv.pkl'), 'wb') as file:
    pickle.dump(lista_prob_pred_test,file)

with open(os.path.join(cartella,'conv2dlstm_ytest_cs_cv.pkl'), 'wb') as file:
    pickle.dump(ytest_tot,file)

with open(os.path.join(cartella,'conv2dlstm_ytrain_cs_cv.pkl'), 'wb') as file:
    pickle.dump(ytrain_tot,file)


with open(os.path.join(cartella,"iperparametri_conv2dlstm_cs_cv.pkl"),"wb") as file:
    pickle.dump(iperparametri, file)


with open(os.path.join(cartella,'conv2dlstm_pred_train_cs_cv.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train,file)

with open(os.path.join(cartella,'conv2dlstm_pred_test_cs_cv.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_test,file)









