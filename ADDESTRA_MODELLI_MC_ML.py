import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sympy.strategies.core import switch
import pickle
from sklearn.metrics import confusion_matrix
from scipy.special import expit

matrice_features = np.load('matrice_features_mc.npy')
matrice_features = matrice_features.transpose(0,2,1).reshape(matrice_features.shape[0], -1)


with open('nome_file_vf2.pkl', 'rb') as f:
    nome_file_vf2 = pickle.load(f)

matrice_feat_train_cs = np.load('matrice_features_train_cs.npy')
matrice_feat_train_cv = np.load('matrice_features_train_cv.npy')
matrice_feat_train_ncs = np.load('matrice_features_train_ncs.npy')
matrice_feat_train_cs_cv = np.load('matrice_features_train_cs_cv.npy')

matrice_feat_test_cs = np.load('matrice_features_test_cs.npy')
matrice_feat_test_cv = np.load('matrice_features_test_cv.npy')
matrice_feat_test_ncs = np.load('matrice_features_test_ncs.npy')
matrice_feat_test_cs_cv = np.load('matrice_features_test_cs_cv.npy')

with open('nome_file_vf2_train_cs.pkl', 'rb') as file:
    nome_train_cs = pickle.load(file)

with open('nome_file_vf2_train_ncs.pkl', 'rb') as file:
    nome_train_ncs = pickle.load(file)

with open('nome_file_vf2_train_cs_cv.pkl', 'rb') as file:
    nome_train_cs_cv = pickle.load(file)

with open('nome_file_vf2_train_cv.pkl', 'rb') as file:
    nome_train_cv = pickle.load(file)

with open('nome_file_vf2_test_cs.pkl', 'rb') as file:
    nome_test_cs = pickle.load(file)

with open('nome_file_vf2_test_ncs.pkl', 'rb') as file:
    nome_test_ncs = pickle.load(file)

with open('nome_file_vf2_test_cs_cv.pkl', 'rb') as file:
    nome_test_cs_cv = pickle.load(file)

with open('nome_file_vf2_test_cv.pkl', 'rb') as file:
    nome_test_cv = pickle.load(file)

# trasformo la forma della matrice da 2200*4*1050 a 2200*1050*4 e poi faccio il reshape così da mantenere il raggruppamento per feature
matrice_feat_train_cs = matrice_feat_train_cs.transpose(0,2,1).reshape(matrice_feat_train_cs.shape[0],-1)
matrice_feat_train_cv = matrice_feat_train_cv.transpose(0,2,1).reshape(matrice_feat_train_cv.shape[0],-1)
matrice_feat_train_ncs = matrice_feat_train_ncs.transpose(0,2,1).reshape(matrice_feat_train_ncs.shape[0],-1)
matrice_feat_train_cs_cv = matrice_feat_train_cs_cv.transpose(0,2,1).reshape(matrice_feat_train_cs_cv.shape[0],-1)

matrice_feat_test_cs = matrice_feat_test_cs.transpose(0,2,1).reshape(matrice_feat_test_cs.shape[0],-1)
matrice_feat_test_cv = matrice_feat_test_cv.transpose(0,2,1).reshape(matrice_feat_test_cv.shape[0],-1)
matrice_feat_test_ncs = matrice_feat_test_ncs.transpose(0,2,1).reshape(matrice_feat_test_ncs.shape[0],-1)
matrice_feat_test_cs_cv = matrice_feat_test_cs_cv.transpose(0,2,1).reshape(matrice_feat_test_cs_cv.shape[0],-1)


def switch_train(train):
    if train == "train_ncs":
        return matrice_feat_train_ncs

    elif train == "train_cs":
        return matrice_feat_train_cs

    elif train == "train_cv":
        return matrice_feat_train_cv

    elif train == "train_cs_cv":
        return matrice_feat_train_cs_cv

x_train = switch_train("train_ncs")

def switch_test(test):
    if test == "test_ncs":
        return matrice_feat_test_ncs

    elif test == "test_cs":
        return matrice_feat_test_cs

    elif test == "test_cv":
        return matrice_feat_test_cv

    elif test == "test_cs_cv":
        return matrice_feat_test_cs_cv

x_test = switch_test("test_ncs")

def classe_train(nome_train):

    classe = []
    for nome in nome_train:
        if "A007" in nome:
            y = 1
        elif "A008" in nome:
            y = 2
        elif "A009" in nome:
            y = 3
        elif "A027" in nome:
            y = 4
        elif "A042" in nome:
            y = 5
        elif "A043" in nome:
            y = 6
        elif "A046" in nome:
            y = 7
        elif "A047" in nome:
            y = 8
        elif "A054" in nome:
            y = 9
        elif "A059" in nome:
            y = 10
        elif "A060" in nome:
            y = 11
        elif "A069" in nome:
            y = 12
        elif "A070" in nome:
            y = 13
        elif "A080" in nome:
            y = 14
        elif "A099" in nome:
            y = 15

        classe.append(y)

    return classe

y_train = classe_train(nome_train_ncs)



def classe_test(nome_test):

    classe = []
    for nome in nome_test:
        if "A007" in nome:
            y = 1
        elif "A008" in nome:
            y = 2
        elif "A009" in nome:
            y = 3
        elif "A027" in nome:
            y = 4
        elif "A042" in nome:
            y = 5
        elif "A043" in nome:
            y = 6
        elif "A046" in nome:
            y = 7
        elif "A047" in nome:
            y = 8
        elif "A054" in nome:
            y = 9
        elif "A059" in nome:
            y = 10
        elif "A060" in nome:
            y = 11
        elif "A069" in nome:
            y = 12
        elif "A070" in nome:
            y = 13
        elif "A080" in nome:
            y = 14
        elif "A099" in nome:
            y = 15

        classe.append(y)

    return classe

y_test = classe_test(nome_test_ncs)

class Addestramento_modelli_ML:
    def __init__(self,x_train,x_test,y_train):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

    def modello_KNN(self):
        parametri = [5, 'euclidean']
        knn_pipeline = Pipeline([('scaler', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=parametri[0],metric = parametri[1]))])
        #knn = KNeighborsClassifier(n_neighbors=parametri[0],metric = parametri[1])
        knn_pipeline.fit(self.x_train,self.y_train)
        predizioni = knn_pipeline.predict(self.x_test)

        return predizioni

    def modello_SVM(self):

        parametri = [0.75, "rbf"]
        svm_pipeline = Pipeline([('scaler',StandardScaler()),('svm', SVC(C=parametri[0], kernel=parametri[1]))])
        #svm = SVC(C=parametri[0], kernel=parametri[1])
        svm_pipeline.fit(self.x_train,self.y_train)
        predizioni = svm_pipeline.predict(self.x_test)

        return predizioni

    def modello_RF(self):

        parametri = [300,"gini",3,5,"sqrt"]
        rf_pipeline = Pipeline([('scaler',StandardScaler()),('rf', RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf = parametri[2],min_samples_split = parametri[3],max_features=parametri[4]))])
        #rf = RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf = parametri[2],min_samples_split = parametri[3],max_features=parametri[4])
        rf_pipeline.fit(self.x_train,self.y_train)
        predizioni = rf_pipeline.predict(self.x_test)

        return predizioni

    def modello_ADABOOST(self):

        parametri = [DecisionTreeClassifier(max_depth = 3),200,1]
        ab_pipeline = Pipeline([('scaler',StandardScaler()),('ab', AdaBoostClassifier(estimator=parametri[0],n_estimators=parametri[1],learning_rate=parametri[2], algorithm="SAMME"))])
        ab_pipeline.fit(self.x_train, self.y_train)
        predizioni = ab_pipeline.predict(self.x_test)

        return predizioni

    def modello_MLP(self):

        parametri = [(128,),0.001,500] # neuroni dello strato nascosto e learning rate init
        mlp_pipeline = Pipeline([('scaler',StandardScaler()),('mlp', MLPClassifier(hidden_layer_sizes=parametri[0],learning_rate_init=parametri[1],max_iter=parametri[2]))])
        mlp_pipeline.fit(self.x_train,self.y_train)
        predizioni = mlp_pipeline.predict(self.x_test)

        return predizioni

#algoritmi_ml = Addestramento_modelli_ML(x_train,x_test,y_train)
#predizioni_knn = algoritmi_ml.modello_KNN()
#predizioni_svm = algoritmi_ml.modello_SVM()
#predizioni_rf = algoritmi_ml.modello_RF()
#predizioni_ab = algoritmi_ml.modello_ADABOOST()
#predizioni_mlp = algoritmi_ml.modello_MLP()

#accuratezza_knn = accuracy_score(y_test, predizioni_knn)
#accuratezza_rf = accuracy_score(y_test, predizioni_rf)
#accuratezza_svm = accuracy_score(y_test, predizioni_svm)
#accuratezza_ab = accuracy_score(y_test, predizioni_ab)
#accuratezza_mlp = accuracy_score(y_test, predizioni_mlp)

#print(f"accuratezza rf",accuratezza_rf)
#print(f"accuratezza svm", accuratezza_svm)
#print(f"accuratezza knn", accuratezza_knn)
#print(f"accuratezza ab", accuratezza_ab)
#print(f"accuratezza mlp", accuratezza_mlp)


# adesso faccio 100 simulazioni e ogni volta calcolo l'accuratezza


confusion_matrix_knn = np.zeros((15,15),dtype=int)
confusion_matrix_svm = np.zeros((15,15),dtype=int)
confusion_matrix_rf = np.zeros((15,15),dtype=int)
confusion_matrix_ab = np.zeros((15,15),dtype=int)
confusion_matrix_mlp = np.zeros((15,15),dtype=int)


lista_acc_knn_test = []
lista_acc_svm_test = []
lista_acc_rf_test = []
lista_acc_ab_test = []
lista_acc_mlp_test = []

lista_acc_knn_train = []
lista_acc_svm_train = []
lista_acc_rf_train = []
lista_acc_ab_train = []
lista_acc_mlp_train = []

lista_f1_knn_test = []
lista_f1_svm_test = []
lista_f1_rf_test = []
lista_f1_ab_test = []
lista_f1_mlp_test = []

lista_f1_knn_train = []
lista_f1_svm_train = []
lista_f1_rf_train = []
lista_f1_ab_train = []
lista_f1_mlp_train = []

dim_train_sim = []
dim_test_sim = []
etichetta_is_sub_tot = []
etichetta_is_sett_tot =  []
etichetta_is_tot_train = []
etichetta_is_tot_test = []
lista_nomi_sim = []
id_sim = []

lista_predizioni_train_knn = []
lista_predizioni_train_svm = []
lista_predizioni_train_rf = []
lista_predizioni_train_ab = []
lista_predizioni_train_mlp = []

lista_predizioni_test_knn = []
lista_predizioni_test_svm = []
lista_predizioni_test_rf = []
lista_predizioni_test_ab = []
lista_predizioni_test_mlp = []


is_pred_cc_knn_test_tot = []
is_pred_cc_svm_test_tot = []
is_pred_cc_rf_test_tot = []
is_pred_cc_ab_test_tot = []
is_pred_cc_mlp_test_tot = []

is_pred_cc_knn_train_tot = []
is_pred_cc_svm_train_tot = []
is_pred_cc_rf_train_tot = []
is_pred_cc_ab_train_tot = []
is_pred_cc_mlp_train_tot = []

item_loss_knn_test_tot = []
item_loss_svm_test_tot = []
item_loss_rf_test_tot = []
item_loss_ab_test_tot = []
item_loss_mlp_test_tot = []

item_loss_knn_train_tot = []
item_loss_svm_train_tot = []
item_loss_rf_train_tot = []
item_loss_ab_train_tot = []
item_loss_mlp_train_tot = []

prob_knn_train_tot = []
prob_svm_train_tot = []
prob_rf_train_tot = []
prob_ab_train_tot = []
prob_mlp_train_tot = []

prob_knn_test_tot = []
prob_svm_test_tot = []
prob_rf_test_tot = []
prob_ab_test_tot = []
prob_mlp_test_tot = []

ytest_tot = []
ytrain_tot = []


modelli_knn = []
modelli_svm = []
modelli_rf = []
modelli_ab = []
modelli_mlp = []

i = 0
for i in range(100):

    def switch_type(type_analisi,matrice,nome_file_vf2):
        if type_analisi == "analisi_cv":

            indici = []
            for index, value in enumerate(nome_file_vf2):
                indici.append(index)

            lista_nomi = []
            for nome in nome_file_vf2:
                lista_nomi.append(nome)

            etichetta_tot_sub = []
            for j in range(len(indici)):
                etich = "FALSE"
                etichetta_tot_sub.append(etich)

            etichetta_tot_sett = []
            for j in range(len(indici)):
                etich = "TRUE"
                etichetta_tot_sett.append(etich)


            # faccio una suddivisione cross setting (cross view)

            def estrazione_set(stringa):
                inizio = stringa.index("S")
                fine = inizio + 4
                return stringa[inizio:fine]

            setting = []
            for nome in nome_file_vf2:
                substring = estrazione_set(nome)
                setting.append(substring)

            lista_set = sorted(set(setting))

            set_train = random.sample(lista_set, k=23)

            set_test = []

            for setting in lista_set:
                if setting not in set_train:
                    set_test.append(setting)

            set_train = sorted(set_train)
            set_test = sorted(set_test)

            indici_train = []
            indici_test = []

            for setting in set_train:
                for index, nome in enumerate(nome_file_vf2):
                    if setting in nome:
                        indici_train.append(index)

            for setting in set_test:
                for index, nome in enumerate(nome_file_vf2):
                    if setting in nome:
                        indici_test.append(index)

            nome_file_train_vf2 = []
            nome_file_test_vf2 = []

            for i in indici_train:
                nome = nome_file_vf2[i]
                nome_file_train_vf2.append(nome)

            for i in indici_test:
                nome = nome_file_vf2[i]
                nome_file_test_vf2.append(nome)

            data_train_vf2 = matrice[indici_train, :]
            data_test_vf2 = matrice[indici_test, :]

            return data_train_vf2, data_test_vf2, etichetta_tot_sub, etichetta_tot_sett, nome_file_train_vf2, nome_file_test_vf2, lista_nomi


        elif type_analisi == "analisi_ncs":

            lista_nomi = []
            for nome in nome_file_vf2:
                lista_nomi.append(nome)

            etichetta_tot_sub = []
            for j in range(len(nome_file_vf2)):
                etich = "FALSE"
                etichetta_tot_sub.append(etich)

            etichetta_tot_sett = []
            for j in range(len(nome_file_vf2)):
                etich = "FALSE"
                etichetta_tot_sett.append(etich)

            indici = []
            for index, value in enumerate(nome_file_vf2):
                indici.append(index)

            # faccio una suddivisione non cross subj (70/30)

            dim_train = round(70 * len(indici) / 100)
            indici_train = random.sample(indici, dim_train)

            indici_test = []
            for i in indici:
                if i not in indici_train:
                    indici_test.append(i)



            nome_file_vf2_train = []
            for i in indici_train:
                nome = nome_file_vf2[i]
                nome_file_vf2_train.append(nome)

            nome_file_vf2_test = []
            for i in indici_test:
                nome = nome_file_vf2[i]
                nome_file_vf2_test.append(nome)

            data_train_vf2 = matrice[indici_train, :]
            data_test_vf2 = matrice[indici_test, :]


            return data_train_vf2, data_test_vf2, etichetta_tot_sub, etichetta_tot_sett, nome_file_vf2_train, nome_file_vf2_test, lista_nomi

        elif type_analisi == "analisi_cs":

            lista_nomi = []
            for nome in nome_file_vf2:
                lista_nomi.append(nome)

            etichetta_tot_sub = []
            for j in range(len(nome_file_vf2)):
                etich = "TRUE"
                etichetta_tot_sub.append(etich)

            etichetta_tot_sett = []
            for j in range(len(nome_file_vf2)):
                etich = "FALSE"
                etichetta_tot_sett.append(etich)

            def estrazione_paz(stringa):
                inizio = stringa.index("P")
                fine = inizio + 4
                return stringa[inizio:fine]

            pazienti = []
            for nome in nome_file_vf2:
                substring = estrazione_paz(nome)
                pazienti.append(substring)

            lista_paz = sorted(set(pazienti))

            pazienti_train = random.sample(lista_paz, k=75)

            pazienti_test = []

            for paz in lista_paz:
                if paz not in pazienti_train:
                    pazienti_test.append(paz)

            pazienti_train = sorted(pazienti_train)
            pazienti_test = sorted(pazienti_test)

            index_train = []
            for paz in pazienti_train:
                for index, nome in enumerate(nome_file_vf2):
                    if paz in nome:
                        index_train.append(index)

            index_test = []
            for paz in pazienti_test:
                for index, nome in enumerate(nome_file_vf2):
                    if paz in nome:
                        index_test.append(index)


            nome_file_vf2_train = []
            for i in index_train:
                nome = nome_file_vf2[i]
                nome_file_vf2_train.append(nome)

            nome_file_vf2_test = []
            for i in index_test:
                nome = nome_file_vf2[i]
                nome_file_vf2_test.append(nome)

            data_train_vf2 = matrice[index_train, :]
            data_test_vf2 = matrice[index_test, :]


            return data_train_vf2, data_test_vf2, etichetta_tot_sub, etichetta_tot_sett, nome_file_vf2_train, nome_file_vf2_test, lista_nomi

        elif type_analisi == "analisi_cs_cv":

            lista_nomi = []
            for nome in nome_file_vf2:
                lista_nomi.append(nome)

            def estrazione_paz(stringa):
                inizio = stringa.index("P")
                fine = inizio + 4
                return stringa[inizio:fine]

            pazienti = []
            for nome in nome_file_vf2:
                substring = estrazione_paz(nome)
                pazienti.append(substring)

            lista_paz = sorted(set(pazienti))

            pazienti_train = random.sample(lista_paz, k=70)

            pazienti_test = []

            for paz in lista_paz:
                if paz not in pazienti_train:
                    pazienti_test.append(paz)

            pazienti_train = sorted(pazienti_train)
            pazienti_test = sorted(pazienti_test)

            index_train = []
            for paz in pazienti_train:
                for index, nome in enumerate(nome_file_vf2):
                    if paz in nome:
                        index_train.append(index)

            index_test = []
            for paz in pazienti_test:
                for index, nome in enumerate(nome_file_vf2):
                    if paz in nome:
                        index_test.append(index)

            nome_file_vf2_train_cs = []
            for i in index_train:
                nome = nome_file_vf2[i]
                nome_file_vf2_train_cs.append(nome)

            nome_file_vf2_test_cs = []
            for i in index_test:
                nome = nome_file_vf2[i]
                nome_file_vf2_test_cs.append(nome)

            data_train_vf2_cs = matrice[index_train, :]
            data_test_vf2_cs = matrice[index_test, :]

            # faccio una suddivisione cross view

            def estrazione_set(stringa):
                inizio = stringa.index("S")
                fine = inizio + 4
                return stringa[inizio:fine]

            setting = []
            for nome in nome_file_vf2:
                substring = estrazione_set(nome)
                setting.append(substring)

            lista_set = sorted(set(setting))

            set_train = random.sample(lista_set, k=20)

            set_test = []

            for setting in lista_set:
                if setting not in set_train:
                    set_test.append(setting)

            set_train = sorted(set_train)
            set_test = sorted(set_test)

            indici_train_cs_cv = []
            indici_test_cs_cv = []

            for setting in set_train:
                for index, nome in enumerate(nome_file_vf2_train_cs):
                    if setting in nome:
                        indici_train_cs_cv.append(index)

            for setting in set_test:
                for index, nome in enumerate(nome_file_vf2_test_cs):
                    if setting in nome:
                        indici_test_cs_cv.append(index)

            data_train_vf2_cs_cv = data_train_vf2_cs[indici_train_cs_cv, :]
            data_test_vf2_cs_cv = data_test_vf2_cs[indici_test_cs_cv, :]

            nome_file_vf2_train_cs_cv = []
            for i in indici_train_cs_cv:
                nome = nome_file_vf2_train_cs[i]
                nome_file_vf2_train_cs_cv.append(nome)

            nome_file_vf2_test_cs_cv = []
            for i in indici_test_cs_cv:
                nome = nome_file_vf2_test_cs[i]
                nome_file_vf2_test_cs_cv.append(nome)

            dim = len(nome_file_vf2_train_cs_cv) + len(nome_file_vf2_test_cs_cv)
            etichetta_tot_sub = []
            for j in range(dim):
                etich = "TRUE"
                etichetta_tot_sub.append(etich)

            etichetta_tot_sett = []
            for j in range(dim):
                etich = "TRUE"
                etichetta_tot_sett.append(etich)

            return data_train_vf2_cs_cv, data_test_vf2_cs_cv, etichetta_tot_sub, etichetta_tot_sett, nome_file_vf2_train_cs_cv, nome_file_vf2_test_cs_cv, lista_nomi


    [x_train,x_test,etichetta_tot_sub, etichetta_tot_sett, nome_vf2_train,nome_vf2_test, lista_nomi] = switch_type('analisi_cs_cv',matrice_features,nome_file_vf2)

    dim_train = len(x_train)
    dim_test = len(x_test)
    dim_tot = dim_train + dim_test

    vettore = np.full(dim_tot, dim_train)
    dim_train_sim.append(vettore)

    vettore = np.full(dim_tot, dim_test)
    dim_test_sim.append(vettore)

    vettore = np.full(dim_tot, i)
    id_sim.append(vettore)

    etichetta_is_sub_tot.append(etichetta_tot_sub)
    etichetta_is_sett_tot.append(etichetta_tot_sett)
    etichetta_is_tot_train.append(nome_vf2_train)
    etichetta_is_tot_test.append(nome_vf2_test)

    lista_nomi_sim.append(lista_nomi)

    def classe_train(nome_vf2_train):

        classe = []
        for nome in nome_vf2_train:
            if "A007" in nome:
                y = "A007"
            elif "A008" in nome:
                y = "A008"
            elif "A009" in nome:
                y = "A009"
            elif "A027" in nome:
                y = "A027"
            elif "A042" in nome:
                y = "A042"
            elif "A043" in nome:
                y = "A043"
            elif "A046" in nome:
                y = "A046"
            elif "A047" in nome:
                y = "A047"
            elif "A054" in nome:
                y = "A054"
            elif "A059" in nome:
                y = "A059"
            elif "A060" in nome:
                y = "A060"
            elif "A069" in nome:
                y = "A069"
            elif "A070" in nome:
                y = "A070"
            elif "A080" in nome:
                y = "A080"
            elif "A099" in nome:
                y = "A099"

            classe.append(y)

        return classe


    y_train = classe_train(nome_vf2_train)



    def classe_test(nome_vf2_test):

        classe = []
        for nome in nome_vf2_test:
            if "A007" in nome:
                y = "A007"
            elif "A008" in nome:
                y = "A008"
            elif "A009" in nome:
                y = "A009"
            elif "A027" in nome:
                y = "A027"
            elif "A042" in nome:
                y = "A042"
            elif "A043" in nome:
                y = "A043"
            elif "A046" in nome:
                y = "A046"
            elif "A047" in nome:
                y = "A047"
            elif "A054" in nome:
                y = "A054"
            elif "A059" in nome:
                y = "A059"
            elif "A060" in nome:
                y = "A060"
            elif "A069" in nome:
                y = "A069"
            elif "A070" in nome:
                y = "A070"
            elif "A080" in nome:
                y = "A080"
            elif "A099" in nome:
                y = "A099"

            classe.append(y)

        return classe


    y_test = classe_test(nome_vf2_test)


    ytest_tot.append(y_test)
    ytrain_tot.append(y_train)



    class Addestra_modelli_ML:
        def __init__(self, x_train, x_test, y_train):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train

        def modello_KNN(self):
            parametri = [7, 'euclidean']
            knn_pipeline = Pipeline([('scaler', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=parametri[0], metric=parametri[1]))])
            # knn = KNeighborsClassifier(n_neighbors=parametri[0],metric = parametri[1])
            knn_pipeline.fit(self.x_train, self.y_train)
            predizioni_train = knn_pipeline.predict(self.x_train)
            predizioni_test = knn_pipeline.predict(self.x_test)
            prob_train = knn_pipeline.predict_proba(self.x_train)
            prob_test = knn_pipeline.predict_proba(self.x_test)

            return predizioni_test, predizioni_train, prob_train, prob_test, knn_pipeline

        def modello_SVM(self):
            parametri = [0.75, "rbf"]
            svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=parametri[0], kernel=parametri[1]))])
            # svm = SVC(C=parametri[0], kernel=parametri[1])
            svm_pipeline.fit(self.x_train, self.y_train)
            predizioni_train = svm_pipeline.predict(self.x_train)
            predizioni_test = svm_pipeline.predict(self.x_test)

            decision_scores = svm_pipeline.decision_function(self.x_test)
            y_prob_test = expit(decision_scores)
            prob_test = np.stack([1 - y_prob_test, y_prob_test], -1)

            decision_scores = svm_pipeline.decision_function(self.x_train)
            y_prob_train = expit(decision_scores)
            prob_train = np.stack([1 - y_prob_train, y_prob_train], -1)

            return predizioni_test, predizioni_train,prob_train, prob_test, svm_pipeline

        def modello_RF(self):
            parametri = [300, "gini", 3, 5, "sqrt"]
            rf_pipeline = Pipeline([('scaler', StandardScaler()), ('rf',RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf=parametri[2],min_samples_split=parametri[3],max_features=parametri[4]))])
            # rf = RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf = parametri[2],min_samples_split = parametri[3],max_features=parametri[4])
            rf_pipeline.fit(self.x_train, self.y_train)
            predizioni_train = rf_pipeline.predict(self.x_train)
            predizioni_test = rf_pipeline.predict(self.x_test)
            prob_train = rf_pipeline.predict_proba(self.x_train)
            prob_test = rf_pipeline.predict_proba(self.x_test)

            return predizioni_test, predizioni_train,prob_train, prob_test, rf_pipeline

        def modello_ADABOOST(self):
            parametri = [DecisionTreeClassifier(max_depth=3), 200, 1]
            ab_pipeline = Pipeline([('scaler', StandardScaler()), ('ab', AdaBoostClassifier(estimator=parametri[0],n_estimators=parametri[1],learning_rate=parametri[2],algorithm="SAMME"))])
            ab_pipeline.fit(self.x_train, self.y_train)
            predizioni_train = ab_pipeline.predict(self.x_train)
            predizioni_test = ab_pipeline.predict(self.x_test)
            prob_train = ab_pipeline.predict_proba(self.x_train)
            prob_test = ab_pipeline.predict_proba(self.x_test)

            return predizioni_test, predizioni_train,prob_train, prob_test, ab_pipeline

        def modello_MLP(self):
            parametri = [(128,), 0.001, 500]  # neuroni dello strato nascosto e learning rate init
            mlp_pipeline = Pipeline([('scaler', StandardScaler()), ('mlp',MLPClassifier(hidden_layer_sizes=parametri[0],learning_rate_init=parametri[1],max_iter=parametri[2]))])
            mlp_pipeline.fit(self.x_train, self.y_train)
            predizioni_train = mlp_pipeline.predict(self.x_train)
            predizioni_test = mlp_pipeline.predict(self.x_test)
            prob_train = mlp_pipeline.predict_proba(self.x_train)
            prob_test = mlp_pipeline.predict_proba(self.x_test)

            return predizioni_test, predizioni_train,prob_train, prob_test, mlp_pipeline


    algoritmi_ml = Addestra_modelli_ML(x_train, x_test, y_train)
    [predizioni_knn_test, predizioni_knn_train, prob_knn_train, prob_knn_test, knn_pipeline] = algoritmi_ml.modello_KNN()
    [predizioni_svm_test, predizioni_svm_train, prob_svm_train, prob_svm_test, svm_pipeline] = algoritmi_ml.modello_SVM()
    [predizioni_rf_test, predizioni_rf_train, prob_rf_train, prob_rf_test, rf_pipeline] = algoritmi_ml.modello_RF()
    [predizioni_ab_test, predizioni_ab_train, prob_ab_train, prob_ab_test, ab_pipeline] = algoritmi_ml.modello_ADABOOST()
    [predizioni_mlp_test, predizioni_mlp_train, prob_mlp_train, prob_mlp_test, mlp_pipeline] = algoritmi_ml.modello_MLP()


    modelli_knn.append(knn_pipeline)
    modelli_svm.append(svm_pipeline)
    modelli_rf.append(rf_pipeline)
    modelli_ab.append(ab_pipeline)
    modelli_mlp.append(mlp_pipeline)


    prob_knn_train_tot.append(prob_knn_train)
    prob_svm_train_tot.append(prob_svm_train)
    prob_rf_train_tot.append(prob_rf_train)
    prob_ab_train_tot.append(prob_ab_train)
    prob_mlp_train_tot.append(prob_mlp_train)

    prob_knn_test_tot.append(prob_knn_test)
    prob_svm_test_tot.append(prob_svm_test)
    prob_rf_test_tot.append(prob_rf_test)
    prob_ab_test_tot.append(prob_ab_test)
    prob_mlp_test_tot.append(prob_mlp_test)


    is_pred_cc_knn_test = []
    is_pred_cc_svm_test = []
    is_pred_cc_rf_test = []
    is_pred_cc_ab_test = []
    is_pred_cc_mlp_test = []

    item_loss_knn_test = []
    item_loss_svm_test = []
    item_loss_rf_test = []
    item_loss_ab_test = []
    item_loss_mlp_test = []

    item_loss_knn_train = []
    item_loss_svm_train = []
    item_loss_rf_train = []
    item_loss_ab_train = []
    item_loss_mlp_train = []

    is_pred_cc_knn_train = []
    is_pred_cc_svm_train = []
    is_pred_cc_rf_train = []
    is_pred_cc_ab_train = []
    is_pred_cc_mlp_train = []

    for j in range(len(y_test)):
        if y_test[j] == predizioni_knn_test[j]:
            is_pred = 1
            loss = 0
        elif y_test[j] != predizioni_knn_test[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_knn_test.append(is_pred)
        item_loss_knn_test.append(loss)


    is_pred_cc_knn_test_tot.append(is_pred_cc_knn_test)
    item_loss_knn_test_tot.append(item_loss_knn_test)


    for j in range(len(y_test)):
        if y_test[j] == predizioni_svm_test[j]:
            is_pred = 1
            loss = 0
        elif y_test[j] != predizioni_svm_test[j]:
            is_pred_ = 0
            loss = 1

        is_pred_cc_svm_test.append(is_pred)
        item_loss_svm_test.append(loss)

    is_pred_cc_svm_test_tot.append(is_pred_cc_svm_test)
    item_loss_svm_test_tot.append(item_loss_svm_test)

    for j in range(len(y_test)):
        if y_test[j] == predizioni_rf_test[j]:
            is_pred = 1
            loss = 0
        elif y_test[j] != predizioni_rf_test[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_rf_test.append(is_pred)
        item_loss_rf_test.append(loss)

    is_pred_cc_rf_test_tot.append(is_pred_cc_rf_test)
    item_loss_rf_test_tot.append(item_loss_rf_test)

    for j in range(len(y_test)):
        if y_test[j] == predizioni_ab_test[j]:
            is_pred = 1
            loss = 0
        elif y_test[j] != predizioni_ab_test[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_ab_test.append(is_pred)
        item_loss_ab_test.append(loss)

    is_pred_cc_ab_test_tot.append(is_pred_cc_ab_test)
    item_loss_ab_test_tot.append(item_loss_ab_test)

    for j in range(len(y_test)):
        if y_test[j] == predizioni_mlp_test[j]:
            is_pred = 1
            loss = 0
        elif y_test[j] != predizioni_mlp_test[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_mlp_test.append(is_pred)
        item_loss_mlp_test.append(loss)


    is_pred_cc_mlp_test_tot.append(is_pred_cc_mlp_test)
    item_loss_mlp_test_tot.append(item_loss_mlp_test)


    for j in range(len(y_train)):
        if y_train[j] == predizioni_knn_train[j]:
            is_pred = 1
            loss = 0
        elif y_train[j] != predizioni_svm_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_knn_train.append(is_pred)
        item_loss_knn_train.append(loss)

    is_pred_cc_knn_train_tot.append(is_pred_cc_knn_train)
    item_loss_knn_train_tot.append(item_loss_knn_train)

    for j in range(len(y_train)):
        if y_train[j] == predizioni_svm_train[j]:
            is_pred = 1
            loss = 0
        elif y_train[j] != predizioni_svm_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_svm_train.append(is_pred)
        item_loss_svm_train.append(loss)

    is_pred_cc_svm_train_tot.append(is_pred_cc_svm_train)
    item_loss_svm_train_tot.append(item_loss_svm_train)


    for j in range(len(y_train)):
        if y_train[j] == predizioni_rf_train[j]:
            is_pred = 1
            loss = 0
        elif y_train[j] != predizioni_rf_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_rf_train.append(is_pred)
        item_loss_rf_train.append(loss)

    is_pred_cc_rf_train_tot.append(is_pred_cc_rf_train)
    item_loss_rf_train_tot.append(item_loss_rf_train)

    for j in range(len(y_train)):
        if y_train[j] == predizioni_ab_train[j]:
            is_pred = 1
            loss = 0
        elif y_train[j] != predizioni_ab_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_ab_train.append(is_pred)
        item_loss_ab_train.append(loss)

    is_pred_cc_ab_train_tot.append(is_pred_cc_ab_train)
    item_loss_ab_train_tot.append(item_loss_ab_train)

    for j in range(len(y_train)):
        if y_train[j] == predizioni_mlp_train[j]:
            is_pred = 1
            loss = 0
        elif y_train[j] != predizioni_mlp_train[j]:
            is_pred = 0
            loss = 1

        is_pred_cc_mlp_train.append(is_pred)
        item_loss_mlp_train.append(loss)


    is_pred_cc_mlp_train_tot.append(is_pred_cc_mlp_train)
    item_loss_mlp_train_tot.append(item_loss_mlp_train)


    lista_predizioni_train_knn.append(predizioni_knn_train)
    lista_predizioni_train_svm.append(predizioni_svm_train)
    lista_predizioni_train_rf.append(predizioni_rf_train)
    lista_predizioni_train_ab.append(predizioni_ab_train)
    lista_predizioni_train_mlp.append(predizioni_mlp_train)

    lista_predizioni_test_knn.append(predizioni_knn_test)
    lista_predizioni_test_svm.append(predizioni_svm_test)
    lista_predizioni_test_rf.append(predizioni_rf_test)
    lista_predizioni_test_ab.append(predizioni_ab_test)
    lista_predizioni_test_mlp.append(predizioni_mlp_test)

    accuratezza_knn_test = accuracy_score(y_test, predizioni_knn_test)
    accuratezza_rf_test = accuracy_score(y_test, predizioni_rf_test)
    accuratezza_svm_test = accuracy_score(y_test, predizioni_svm_test)
    accuratezza_ab_test = accuracy_score(y_test, predizioni_ab_test)
    accuratezza_mlp_test = accuracy_score(y_test, predizioni_mlp_test)

    lista_acc_knn_test.append(accuratezza_knn_test)
    lista_acc_svm_test.append(accuratezza_svm_test)
    lista_acc_rf_test.append(accuratezza_rf_test)
    lista_acc_ab_test.append(accuratezza_ab_test)
    lista_acc_mlp_test.append(accuratezza_mlp_test)

    accuratezza_knn_train = accuracy_score(y_train, predizioni_knn_train)
    accuratezza_rf_train = accuracy_score(y_train, predizioni_rf_train)
    accuratezza_svm_train = accuracy_score(y_train, predizioni_svm_train)
    accuratezza_ab_train = accuracy_score(y_train, predizioni_ab_train)
    accuratezza_mlp_train = accuracy_score(y_train, predizioni_mlp_train)

    lista_acc_knn_train.append(accuratezza_knn_train)
    lista_acc_svm_train.append(accuratezza_svm_train)
    lista_acc_rf_train.append(accuratezza_rf_train)
    lista_acc_ab_train.append(accuratezza_ab_train)
    lista_acc_mlp_train.append(accuratezza_mlp_train)

    f1_knn_test = f1_score(y_test, predizioni_knn_test, average = 'weighted')
    f1_rf_test = f1_score(y_test, predizioni_rf_test, average = 'weighted')
    f1_svm_test = f1_score(y_test, predizioni_svm_test, average = 'weighted')
    f1_ab_test = f1_score(y_test, predizioni_ab_test, average = 'weighted')
    f1_mlp_test = f1_score(y_test, predizioni_mlp_test, average = 'weighted')

    lista_f1_knn_test.append(f1_knn_test)
    lista_f1_svm_test.append(f1_svm_test)
    lista_f1_rf_test.append(f1_rf_test)
    lista_f1_ab_test.append(f1_ab_test)
    lista_f1_mlp_test.append(f1_mlp_test)

    f1_knn_train = f1_score(y_train, predizioni_knn_train, average = 'weighted')
    f1_rf_train = f1_score(y_train, predizioni_rf_train, average = 'weighted')
    f1_svm_train = f1_score(y_train, predizioni_svm_train, average = 'weighted')
    f1_ab_train = f1_score(y_train, predizioni_ab_train, average = 'weighted')
    f1_mlp_train = f1_score(y_train, predizioni_mlp_train, average = 'weighted')

    lista_f1_knn_train.append(f1_knn_train)
    lista_f1_svm_train.append(f1_svm_train)
    lista_f1_rf_train.append(f1_rf_train)
    lista_f1_ab_train.append(f1_ab_train)
    lista_f1_mlp_train.append(f1_mlp_train)



    classi = ['A007', 'A008', 'A009', 'A027', 'A042', 'A043', 'A046', 'A047', 'A054', 'A059', 'A060', 'A069', 'A070','A080', 'A099']

    confusion_matrix_knn += confusion_matrix(y_test, predizioni_knn_test, labels=classi)
    confusion_matrix_svm += confusion_matrix(y_test, predizioni_svm_test, labels=classi)
    confusion_matrix_rf += confusion_matrix(y_test, predizioni_rf_test, labels=classi)
    confusion_matrix_ab += confusion_matrix(y_test, predizioni_ab_test, labels=classi)
    confusion_matrix_mlp += confusion_matrix(y_test, predizioni_mlp_test, labels=classi)



    print(f"iterazione", i)

confusion_matrix_knn = np.round(confusion_matrix_knn/100)
confusion_matrix_svm = np.round(confusion_matrix_svm/100)
confusion_matrix_rf = np.round(confusion_matrix_rf/100)
confusion_matrix_ab = np.round(confusion_matrix_ab/100)
confusion_matrix_mlp = np.round(confusion_matrix_mlp/100)




with open('predizioni_knn_cv_cs_test.pkl','wb') as file:
    pickle.dump(lista_predizioni_test_knn, file)

with open('predizioni_svm_cv_cs_test.pkl','wb') as file:
    pickle.dump(lista_predizioni_test_svm, file)

with open('predizioni_rf_cv_cs_test.pkl','wb') as file:
    pickle.dump(lista_predizioni_test_rf, file)

with open('predizioni_ab_cv_cs_test.pkl','wb') as file:
    pickle.dump(lista_predizioni_test_ab, file)

with open('predizioni_mlp_cv_cs_test.pkl','wb') as file:
    pickle.dump(lista_predizioni_test_mlp, file)

with open('predizioni_knn_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_train_knn, file)

with open('predizioni_svm_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_train_svm, file)

with open('predizioni_rf_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_train_rf, file)

with open('predizioni_ab_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_train_ab, file)

with open('predizioni_mlp_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_predizioni_train_mlp, file)

with open('accuratezza_knn_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_acc_knn_test,file)

with open('accuratezza_svm_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_acc_svm_test,file)

with open('accuratezza_rf_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_acc_rf_test,file)

with open('accuratezza_ab_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_acc_ab_test,file)

with open('accuratezza_mlp_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_acc_mlp_test,file)

with open('accuratezza_knn_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_acc_knn_train, file)

with open('accuratezza_svm_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_acc_svm_train, file)

with open('accuratezza_rf_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_acc_rf_train, file)

with open('accuratezza_ab_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_acc_ab_train, file)

with open('accuratezza_mlp_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_acc_mlp_train, file)

with open('f1_knn_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_f1_knn_test, file)

with open('f1_svm_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_f1_svm_test, file)

with open('f1_rf_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_f1_rf_test, file)

with open('f1_ab_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_f1_ab_test, file)

with open('f1_mlp_cv_cs_test.pkl', 'wb') as file:
    pickle.dump(lista_f1_mlp_test, file)

with open('f1_knn_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_f1_knn_train, file)

with open('f1_svm_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_f1_svm_train, file)

with open('f1_rf_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_f1_rf_train, file)

with open('f1_ab_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_f1_ab_train, file)

with open('f1_mlp_cv_cs_train.pkl', 'wb') as file:
    pickle.dump(lista_f1_mlp_train, file)

with open('simul_id_cv_cs.pkl', 'wb') as file:
    pickle.dump(id_sim,file)

with open('array_dim_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(dim_train_sim,file)

with open('array_dim_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(dim_test_sim,file)

with open('etichetta_is_sub_cv_cs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_sub_tot, file)

with open('etichetta_is_sett_cv_cs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_sett_tot, file)

with open('etichetta_is_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_tot_train, file)

with open('etichetta_is_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(etichetta_is_tot_test, file)

with open('lista_nomi_simulazione_cv_cs.pkl', 'wb') as file:
    pickle.dump(lista_nomi_sim,file)

with open('is_pred_cc_knn_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_knn_train_tot,file)

with open('is_pred_cc_svm_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_svm_train_tot,file)

with open('is_pred_cc_rf_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_rf_train_tot,file)

with open('is_pred_cc_ab_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_ab_train_tot,file)

with open('is_pred_cc_mlp_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_mlp_train_tot,file)

with open('is_pred_cc_knn_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_knn_test_tot,file)

with open('is_pred_cc_svm_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_svm_test_tot,file)

with open('is_pred_cc_rf_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_rf_test_tot,file)

with open('is_pred_cc_ab_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_ab_test_tot,file)

with open('is_pred_cc_mlp_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(is_pred_cc_mlp_test_tot,file)

with open('item_loss_knn_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_knn_train_tot,file)

with open('item_loss_svm_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_svm_train_tot,file)

with open('item_loss_rf_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_rf_train_tot,file)

with open('item_loss_ab_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_ab_train_tot,file)

with open('item_loss_mlp_train_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_mlp_train_tot,file)

with open('item_loss_knn_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_knn_test_tot,file)

with open('item_loss_svm_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_svm_test_tot,file)

with open('item_loss_rf_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_rf_test_tot,file)

with open('item_loss_ab_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_ab_test_tot,file)

with open('item_loss_mlp_test_cv_cs.pkl', 'wb') as file:
    pickle.dump(item_loss_mlp_test_tot,file)

with open('prob_conf_train_knn_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_knn_train_tot,file)

with open('prob_conf_train_svm_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_svm_train_tot,file)

with open('prob_conf_train_rf_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_rf_train_tot,file)

with open('prob_conf_train_ab_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_ab_train_tot,file)

with open('prob_conf_train_mlp_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_mlp_train_tot,file)

with open('prob_conf_test_knn_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_knn_test_tot,file)

with open('prob_conf_test_svm_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_svm_test_tot,file)

with open('prob_conf_test_rf_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_rf_test_tot,file)

with open('prob_conf_test_ab_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_ab_test_tot,file)

with open('prob_conf_test_mlp_cv_cs.pkl', 'wb') as file:
    pickle.dump(prob_mlp_test_tot,file)

with open('ytest_cv_cs.pkl', 'wb') as file:
    pickle.dump(ytest_tot,file)

with open('ytrain_cv_cs.pkl', 'wb') as file:
    pickle.dump(ytrain_tot,file)

with open('confusion_matrix_knn_cv_cs.pkl', 'wb') as file:
    pickle.dump(confusion_matrix_knn,file)

with open('confusion_matrix_svm_cv_cs.pkl', 'wb') as file:
    pickle.dump(confusion_matrix_svm,file)

with open('confusion_matrix_rf_cv_cs.pkl', 'wb') as file:
    pickle.dump(confusion_matrix_rf,file)

with open('confusion_matrix_ab_cv_cs.pkl', 'wb') as file:
    pickle.dump(confusion_matrix_ab, file)

with open( 'confusion_matrix_mlp_cv_cs.pkl', 'wb') as file:
    pickle.dump(confusion_matrix_mlp, file)












