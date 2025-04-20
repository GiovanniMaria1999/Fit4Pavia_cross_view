import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import torch.utils.data as data
import os


with open('dizionario_mc.pkl', 'rb') as file:
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


lista_lengths = []
for seq in dati_skeleton_body:
    lenghts = len(seq)
    lista_lengths.append(lenghts)

max_length = max(lista_lengths)


lista_indici = []
for i in range(len(dati_skeleton_body)):
    lista_indici.append(i)


# faccio il padding del dataset


conf_matrix = np.zeros((15,15),dtype=int)


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

    etichetta_sub_tot = []
    for j in range(len(lista_indici)):
        etich = "FALSE"
        etichetta_sub_tot.append(etich)

    etichetta_sett_tot = []
    for j in range(len(lista_indici)):
        etich = "FALSE"
        etichetta_sett_tot.append(etich)


    index_test = []

    dim_train = round(70*len(dati_skeleton_body)/100)
    index_train = random.sample(lista_indici, dim_train)

    for idx in lista_indici:
        if idx not in index_train:
            index_test.append(idx)

    random.shuffle(index_test)


    xtrain = []
    xtest = []

    for j in index_train:
        paz = dati_skeleton_body[j]
        xtrain.append(paz)

    for j in index_test:
        paz = dati_skeleton_body[j]
        xtest.append(paz)

    paz_train = []
    paz_test = []
    for j in index_train:
        paz = dati_nome_chiave[j]
        paz_train.append(paz)

    for j in index_test:
        paz = dati_nome_chiave[j]
        paz_test.append(paz)

    dim_train = len(index_train)
    dim_test = len(index_test)

    dim_train_sim.append(dim_train)
    dim_test_sim.append(dim_test)

    etichetta_is_tot.append(etichetta_sub_tot)
    etichetta_is_cv_tot.append(etichetta_sett_tot)
    etichetta_is_tot_train.append(paz_train)
    etichetta_is_tot_test.append(paz_test)
    lista_nomi_sim.append(lista_nomi)


    # Concatenare i dati di training per calcolare media e deviazione standard


    all_data = np.concatenate(xtrain, axis = 0)
    media_globale = np.mean(all_data, axis = 0)
    dev_globale = np.std(all_data, axis = 0)

    xtrain_norm = []
    for seq in xtrain:
        seq_norm = (seq - media_globale)/dev_globale
        xtrain_norm.append(seq_norm)

    xtest_norm = []
    for seq in xtest:
        seq_norm = (seq - media_globale)/dev_globale
        xtest_norm.append(seq_norm)

    xtrain = xtrain_norm
    xtest = xtest_norm

    nome_chiave_train = []
    nome_chiave_test = []

    for j in index_train:
        nome = dati_nome_chiave[j]
        nome_chiave_train.append(nome)

    for j in index_test:
        nome = dati_nome_chiave[j]
        nome_chiave_test.append(nome)

    ytrain = []
    ytest = []

    for nome in nome_chiave_train:
        if "A007" in nome:
            y = 0
        elif "A008" in nome:
            y = 1
        elif "A009" in nome:
            y = 2
        elif "A027" in nome:
            y = 3
        elif "A042" in nome:
            y = 4
        elif "A043" in nome:
            y = 5
        elif "A046" in nome:
            y = 6
        elif "A047" in nome:
            y = 7
        elif "A054" in nome:
            y = 8
        elif "A059" in nome:
            y = 9
        elif "A060" in nome:
            y = 10
        elif "A069" in nome:
            y = 11
        elif "A070" in nome:
            y = 12
        elif "A080" in nome:
            y = 13
        elif "A099" in nome:
            y = 14

        ytrain.append(y)



    for nome in nome_chiave_test:
        if "A007" in nome:
            y = 0
        elif "A008" in nome:
            y = 1
        elif "A009" in nome:
            y = 2
        elif "A027" in nome:
            y = 3
        elif "A042" in nome:
            y = 4
        elif "A043" in nome:
            y = 5
        elif "A046" in nome:
            y = 6
        elif "A047" in nome:
            y = 7
        elif "A054" in nome:
            y = 8
        elif "A059" in nome:
            y = 9
        elif "A060" in nome:
            y = 10
        elif "A069" in nome:
            y = 11
        elif "A070" in nome:
            y = 12
        elif "A080" in nome:
            y = 13
        elif "A099" in nome:
            y = 14

        ytest.append(y)


    lista_len_train = []

# effettuo lo zero padding in modo che tutte le sequenze abbiano la stessa lunghezza

    xtrain_padding = []

    for vet in xtrain:
        padding_size = max_length - len(vet)
        vettore = torch.tensor(vet)
        padding_vet = torch.nn.functional.pad(vettore, (0,0,0, padding_size))
        xtrain_padding.append(padding_vet)

    xtrain = torch.stack(xtrain_padding)
    xtrain = xtrain.to(torch.float32)


    xtest_padding = []

    for vet in xtest:
        padding_size = max_length - len(vet)
        vettore = torch.tensor(vet)
        padding_vet = torch.nn.functional.pad(vettore, (0, 0, 0, padding_size))
        xtest_padding.append(padding_vet)

    xtest = torch.stack(xtest_padding)
    xtest = xtest.to(torch.float32)

    xtrain = xtrain.permute(0,2,1)
    xtest = xtest.permute(0,2,1)



    ytrain = torch.tensor(ytrain)
    ytest = torch.tensor(ytest)

    # costruisco la rete

    xtrain = xtrain.unsqueeze(1)  # aggiungo una nuova dimensione (dim = 1) in posizione 1 in xtrain traslando le altre dimensioni
    xtest = xtest.unsqueeze(1)

    in_channels = 1
    out_channels = 64
    num_classes = 15
    hidden_size = 100


    class CONV2DLSTM(nn.Module):
        def __init__(self, in_channels, out_channels, in_features, num_classes):
            super(CONV2DLSTM, self).__init__()  # inizializzo il costrutto Module

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=2,padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=2, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=2, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.2)

            self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2,bidirectional=False)
            self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

        def forward(self, x):
            x = self.dropout(self.pool1(self.relu(self.conv1(x))))
            x = self.dropout(self.pool2(self.relu(self.conv2(x))))
            x = self.dropout(self.pool3(self.relu(self.conv3(x))))


            x = torch.flatten(x, start_dim=2)
            x = x.permute(0, 2, 1)

            lstm_out, (h_n, c_n) = self.lstm(x)  # calcolo la hidden state e la cell state
            out = lstm_out[:, -1, :]
            out = self.fc(out)

            return out


    model = CONV2DLSTM(in_channels, out_channels, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # provo 0.01 o 0.0001 (rete complessa lr basso
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    batch_size = 64

    train_dataset = data.TensorDataset(xtrain, ytrain)  # Crea un dataset
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    iperparametri = {
        "learning_rate": 0.001,
        "epoche": 100,
        "in_channels": 1,
        "out_channels": 64,
        "batch_size": batch_size,
        "num_classes": 15,
        "hidden_size": 100,
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam"
    }



    epoche = 100

    train_losses = []
    model.train()
    for epoca in range(epoche):
        epoch_loss = 0

        for batch_x, batch_y in train_loader:

            optimizer.zero_grad()

            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(epoch_loss)

        #print(f"Epoca {epoca +1}/{epoche}, Loss: {epoch_loss}")

    # Grafico della Loss
    # plt.plot(range(1, epoche + 1), train_losses, label="Training Loss")
    # plt.xlabel("Epoca")
    # plt.ylabel("Loss")
    # plt.title("Andamento della Loss durante il Training")
    # plt.legend()
    # plt.show()

    model.eval()
    with torch.no_grad():
        train_outputs = model(xtrain)
        prob_train = torch.softmax(train_outputs, dim=1).max(dim=1).values  # ottengo la prob di confidenza
        predicted_classes = torch.argmax(train_outputs, dim=1)

    predizioni_train = np.array(predicted_classes)
    predizioni_train = predizioni_train.flatten()  # trasforma l'array nidificato in un array monodimensionale
    y_train = np.array(ytrain)

    accuratezza_train = accuracy_score(y_train, predizioni_train)
    f1_train = f1_score(y_train, predizioni_train, average='macro')

    lista_predizioni_train.append(predizioni_train)
    lista_prob_pred_train.append(prob_train)

    lista_acc_train.append(accuratezza_train)
    lista_f1_train.append(f1_train)

    is_pred_cc_train = []
    item_loss_train = []

    for j in range(len(y_train)):
        if y_train[j] == predizioni_train[j]:
            is_pred = 1
            loss = 0
        elif y_train[j] != predizioni_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_train.append(is_pred)
        item_loss_train.append(loss)

    is_pred_cc_train_tot.append(is_pred_cc_train)
    item_loss_train_tot.append(item_loss_train)

    model.eval()
    with torch.no_grad():

        outputs = model(xtest)
        prob_test = torch.softmax(outputs, dim=1).max(dim=1).values
        predicted_classes = torch.argmax(outputs, dim=1)

    predizioni_test = np.array(predicted_classes)
    predizioni_test = predizioni_test.flatten()  # creo un array monodomensionale
    y_test = np.array(ytest)

    lista_predizioni_test.append(predizioni_test)
    lista_prob_pred_test.append(prob_test)

    is_pred_cc_test = []
    item_loss_test = []

    for j in range(len(y_test)):
        if y_test[j] == predizioni_test[j]:
            is_pred = 1
            loss = 0
        elif y_test[j] != predizioni_test[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_test.append(is_pred)
        item_loss_test.append(loss)

    is_pred_cc_test_tot.append(is_pred_cc_test)
    item_loss_test_tot.append(item_loss_test)

    accuratezza_test = accuracy_score(ytest, predizioni_test)
    f1_test = f1_score(ytest, predizioni_test, average='macro')
    print(f"accuratezza", accuratezza_test, f"f1", f1_test)
    lista_acc_test.append(accuratezza_test)
    lista_f1_test.append(f1_test)
    conf_matrix += confusion_matrix(y_test, predizioni_test)



conf_matrix = np.round(conf_matrix / 100)



vettore_tot = []
for i in range(100):
    vettore = np.full(2371, i)
    vettore_tot.append(vettore)

array_simul_id = np.array(vettore_tot)
array_simul_id = array_simul_id.reshape(-1, 1)  # ottengo un vettore colonna con 314*100 righe


vettore_tot = []
for dim in dim_train_sim:
    vettore = np.full(2371, dim)
    vettore_tot.append(vettore)

array_dim_train_id = np.array(vettore_tot)
array_dim_train_id = array_dim_train_id.reshape(-1, 1)

vettore_tot = []
for dim in dim_test_sim:
    vettore = np.full(2371, dim)
    vettore_tot.append(vettore)

array_dim_test_id = np.array(vettore_tot)
array_dim_test_id = array_dim_test_id.reshape(-1, 1)

cartella = "modelli_mc"

with open('accuratezza_conv2dlstm_mc_ncs_test.pkl', 'wb') as file:
    pickle.dump(lista_acc_test,file)

with open('accuratezza_conv2dlstm_mc_ncs_train.pkl', 'wb') as file:
    pickle.dump(lista_acc_train, file)

with open('f1_conv2dlstm_mc_ncs_test.pkl', 'wb') as file:
    pickle.dump(lista_f1_test,file)

with open('f1_conv2dlstm_mc_ncs_train.pkl', 'wb') as file:
    pickle.dump(lista_f1_train, file)

with open('confmatrix_mc_conv2dlstm_ncs.pkl', 'wb') as file:
    pickle.dump(conf_matrix , file)

with open('conv2dlstm_mc_simul_id_ncs.pkl', 'wb') as file:
    pickle.dump(array_simul_id,file)

with open('conv2dlstm_mc_array_dim_train_ncs.pkl', 'wb') as file:
    pickle.dump(array_dim_train_id,file)

with open('conv2dlstm_mc_array_dim_test_ncs.pkl', 'wb') as file:
    pickle.dump(array_dim_test_id,file)

with open('conv2dlstm_mc_etichetta_is_ncs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_tot, file)

with open('conv2dlstm_mc_etichetta_is_ncs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_cv_tot, file)

with open('conv2dlstm_mc_etichetta_is_train_ncs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_tot_train, file)

with open('conv2dlstm_mc_etichetta_is_test_ncs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_tot_test, file)

with open('conv2dlstm_mc_lista_nomi_simulazione_ncs.pkl', 'wb') as file:
    pickle.dump(lista_nomi_sim,file)

with open('conv2dlstm_mc_is_pred_cc_train_ncs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_train_tot,file)

with open('conv2dlstm_mc_is_pred_cc_test_ncs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_test_tot,file)

with open('conv2dlstm_mc_item_loss_test_ncs.pkl', 'wb') as file:
    pickle.dump(item_loss_test_tot,file)

with open('conv2dlstm_mc_item_loss_train_ncs.pkl', 'wb') as file:
    pickle.dump(item_loss_train_tot,file)

with open('conv2dlstm_prob_mc_conf_train_ncs.pkl', 'wb') as file:
    pickle.dump(lista_prob_pred_train,file)

with open('conv2dlstm_mc_prob_conf_test_ncs.pkl', 'wb') as file:
    pickle.dump(lista_prob_pred_test,file)

with open('conv2dlstm_mc_ytest_ncs.pkl', 'wb') as file:
    pickle.dump(ytest_tot,file)

with open('conv2dlstm_mc_ytrain_ncs.pkl', 'wb') as file:
    pickle.dump(ytrain_tot,file)

with open("iperparametri_conv2dlstm_mc_ncs.pkl","wb") as file:
    pickle.dump(iperparametri, file)

with open('conv2dlstm_mc_pred_train_ncs.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_train,file)

with open('conv2dlstm_mc_pred_test_ncs.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_test,file)







