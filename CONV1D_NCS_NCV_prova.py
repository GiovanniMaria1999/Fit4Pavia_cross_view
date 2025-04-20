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

from Studio_Multiclasse.CONV1D_NCS_mc import dati_skeleton

with open('train_ncv_ncs.pkl', 'rb') as file:
    dizionario_train = pickle.load(file)

with open('test_ncv_ncs.pkl', 'rb') as file:
    dizionario_test = pickle.load(file)

with open('dizionario_mc_vf2.pkl', 'rb') as file:
    dizionario_mc_vf2 = pickle.load(file)

dati_skeleton_body = list(dizionario_mc_vf2.values())


xtrain = list(dizionario_train.values())
nome_chiave_train = list(dizionario_train.keys())

xtest = list(dizionario_train.values())
nome_chiave_test = list(dizionario_test.keys())



lista_lengths = []
for seq in dati_skeleton_body:
    lenghts = len(seq)
    lista_lengths.append(lenghts)

max_length = max(lista_lengths)


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

sequence_length = xtrain.shape[2]


def compute_output_size(input_length, kernel_size, stride, padding, pooling_size):
    conv_length = (input_length - kernel_size + 2 * padding) // stride + 1  # Lunghezza dopo convoluzione
    pooled_length = conv_length // pooling_size  # Lunghezza dopo pooling
    return pooled_length


    # Calcolo dinamico del numero di feature per fc1
seq_len_after_conv1 = compute_output_size(sequence_length, kernel_size=3, stride=1, padding=1, pooling_size=2)
seq_len_after_conv2 = compute_output_size(seq_len_after_conv1, kernel_size=3, stride=1, padding=1, pooling_size=2)
seq_len_after_conv3 = compute_output_size(seq_len_after_conv2, kernel_size=3, stride=1, padding=1, pooling_size=2)


in_features = 256 * seq_len_after_conv3  # 256 = out_channels del terzo livello convoluzionale
in_channels = 75
out_channels = 64
num_classes = 15


class CONV1D(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, num_classes):  # definisco il costruttore di CONV1D
        super(CONV1D, self).__init__()  # serve per chiamare il costruttore della classe base(nn.Module) dalla
            # classe personalizzata. E' un metodo per inizializzare correttamente un classe derivata in Python, in particolare
            # quando eredito da classi che fanno parte di librerie esterne; serve per inizializzare il costruttore nn.Module.

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout(self.pool3(self.relu(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)

        return out


    # inizializzo il modello

model = CONV1D(in_channels=in_channels, out_channels=out_channels, in_features=in_features, num_classes=num_classes)
criterion = nn.CrossEntropyLoss() # gestisce internamente lo stato softmax, non ha senso usare la sigmoid
optimizer = optim.Adam(model.parameters(), lr=0.001)


batch_size = 64

train_dataset = data.TensorDataset(xtrain, ytrain)  # Crea un dataset
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # definisco la funzione di attivazione

def objective(trial):
    hidden_size = 2*(2**(trial.suggest_int("hidden_size", 5, 8)))
    num_layers = trial.suggest_int("num_layers", 1, 3)
    learning_rate = 10**(-trial.suggest_int("learning_rate", 2, 7))
    out_features = 2**(trial.suggest_int("out_features", 4, 7))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # optim.SGD e optim.RMSPROB


    epoche = 100

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
            model.eval()
            with torch.no_grad():
                train_outputs = model(xtrain)
                predicted_classes = torch.argmax(train_outputs, dim=1)

            predizioni = np.array(predicted_classes)
            predizioni = predizioni.flatten()  # trasforma l'array nidificato in un array monodimensionale
            y_train = np.array(ytrain)

            accuratezza_train = accuracy_score(y_train, predizioni)

            trial.report(accuratezza_train, epoca) # devo mettere l'accuratezza train
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        model.train()


    model.eval()
    with torch.no_grad():
        train_outputs = model(xtrain)
        predicted_classes = torch.argmax(train_outputs, dim=1)

    predizioni = np.array(predicted_classes)
    predizioni = predizioni.flatten()  # trasforma l'array nidificato in un array monodimensionale
    y_train = np.array(ytrain)

    accuratezza_train = accuracy_score(y_train, predizioni)
    print("accuratezza train",accuratezza_train)
    return accuratezza_train


    model.eval()
    with torch.no_grad():
        outputs = model(xtest)
        predicted_classes = torch.argmax(outputs, dim=1)

    predizioni = np.array(predicted_classes)
    predizioni = predizioni.flatten()  # creo un array monodomensionale
    ytest = np.array(ytest)


    accuratezza = accuracy_score(ytest, predizioni)
    f1 = f1_score(ytest, predizioni, average='weighted')
    print(f"accuratezza test", accuratezza, f"f1 test", f1)
    lista_acc.append(accuratezza)
    lista_f1.append(f1)

    # creo lo studio optuna

study = optuna.create_study(direction = "maximize", sampler = optuna.samplers.TPESampler(), pruner = optuna.pruners.MedianPruner())  # cerco l'accuratezza massima
study.optimize(objective, n_trials = 50) # faccio 20 prove
print("Migliori parametri", study.best_value)

#fig = optuna.visualization.plot_intermediate_values(study) # curva di accuratezza
#fig.show()
#fig = optuna.visualization.plot_optimization_history(study)
#fig.show()
#fig = optuna.visualization.plot_parallel_coordinate(study)
#fig.show()
#fig = optuna.visualization.plot_param_importances(study) # grafico a barre
#fig.show()







