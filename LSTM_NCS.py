import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import torch.utils.data as data
import optuna

with open('dizionario_mc_vf2.pkl', 'rb') as file:
    dizionario = pickle.load(file)


dati_nome_chiave = list(dizionario.keys())
dati_skeleton_body = list(dizionario.values()) # prendo tutti i dati skeleton e li trasformo il liste


lista_lengths = []
for seq in dati_skeleton_body:
    lenghts = len(seq)
    lista_lengths.append(lenghts)

max_length = max(lista_lengths)

lista_indici = []
for i in range(len(dati_skeleton_body)):
    lista_indici.append(i)

# faccio il padding del dataset

lista_acc = []
lista_f1 = []
conf_matrix = np.zeros((15,15),dtype=int)

for i in range(1):


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

    # Concatenare i dati di training per calcolare media e deviazione standard

    all_data = np.concatenate(xtrain, axis=0)
    media_globale = np.mean(all_data, axis=0)
    dev_globale = np.std(all_data, axis=0)

    xtrain_norm = []
    for seq in xtrain:
        seq_norm = (seq - media_globale) / dev_globale
        xtrain_norm.append(seq_norm)

    xtest_norm = []
    for seq in xtest:
        seq_norm = (seq - media_globale) / dev_globale
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

    xtrain_padding = []

    for vet in xtrain:
        padding_size = max_length - len(vet)
        vettore = torch.tensor(vet)
        padding_vet = torch.nn.functional.pad(vettore, (0, 0, 0, padding_size))
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


    ytrain = torch.tensor(ytrain)
    ytest = torch.tensor(ytest)


    # creazione rete LSTM

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, out_features):
            super(LSTM, self).__init__()

            self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers= num_layers, batch_first = True, bidirectional= False) # hidden size numero di neuroni
            self.fc1 = nn.Linear(hidden_size, out_features)
            self.fc2 = nn.Linear(out_features , num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):

            lstm_out, (h_n, c_n) = self.lstm(x)
            out = lstm_out[:,-1,:] # prendo l'ultimo passo temporale (.detach)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            out = torch.softmax(out, dim = 1)

            return out

    batch_size = 64

    train_dataset = data.TensorDataset(xtrain, ytrain)  # Crea un dataset
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # definisco la funzione di attivazione

    def objective(trial):
        hidden_size = 2**(trial.suggest_int("hidden_size", 5, 8, 1))
        num_layers = trial.suggest_int("num_layers", 1, 3, 1)
        learning_rate = 10**(-trial.suggest_int("learning_rate", 2, 7, 1))
        out_features = 2**(trial.suggest_int("out_features", 4, 7, 1))


        input_size = 75
        num_classes = 15
        model = LSTM(input_size, hidden_size, num_layers, num_classes, out_features)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # optim.SGD e optim.RMSPROB


        epoche = 200

        loss_list = []
        model.train()
        for epoca in range(epoche):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(train_loader)
            loss_list.append(epoch_loss)

            print(f"{epoca} Loss: {epoch_loss}")

            if epoca % 10 == 0:
                trial.report("accuratezza_train", epoca) # devo mettere l'accuratezza train
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()


        model.eval()
        with torch.no_grad():
            train_outputs = model(xtrain)
            predicted_classes = torch.argmax(train_outputs, dim=1)

        predizioni = np.array(predicted_classes)
        predizioni = predizioni.flatten()  # trasforma l'array nidificato in un array monodimensionale
        y_train = np.array(ytrain)

        accuratezza_train = accuracy_score(y_train, predizioni)
        #print("train",accuratezza_train)
        return accuratezza_train

    # creo lo studio optuna

    study = optuna.create_study(direction = "maximize", sampler = optuna.samplers.TPESampler(), pruner = optuna.pruners.MedianPruner())  # cerco l'accuratezza massima
    study.optimize(objective, n_trials = 50) # faccio 20 prove
    print("Migliori parametri", study.best_value)

    fig = optuna.visualization.plot_intermediate_values(study) # curva di accuratezza
    fig.show()
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.show()
    fig = optuna.visualization.plot_param_importances(study) # grafico a barre
    fig.show()

    #model.eval()
    #with torch.no_grad():

    #outputs = model(xtest)
    #predicted_classes = torch.argmax(outputs, dim=1)

    #predizioni = np.array(predicted_classes)
    #predizioni = predizioni.flatten()  # creo un array monodomensionale
    #ytest = np.array(ytest)

    #accuratezza = accuracy_score(ytest, predizioni)
    #f1 = f1_score(ytest, predizioni, average='weighted')
    #print(f"accuratezza", accuratezza, f"f1", f1)
    #lista_acc.append(accuratezza)
    #lista_f1.append(f1)
    #conf_matrix += confusion_matrix(ytest, predizioni)

#conf_matrix = np.round(conf_matrix / 100)

#with open('accuratezza_lstm_ncs.pkl', 'wb') as file:
#    pickle.dump(lista_acc, file)

#with open('f1_lstm_ncs.pkl', 'wb') as file:
#    pickle.dump(lista_f1, file)

#with open('conf_matrix_lstm_ncs.pkl', 'wb') as file:
#    pickle.dump(conf_matrix, file)










