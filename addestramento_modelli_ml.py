import pandas as pd
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
import joblib
from scipy.special import expit
import os

Tabella = pd.read_csv("tabella_features.csv")
tabella_train_ncs = pd.read_csv("tabella_train.csv")
tabella_test_ncs = pd.read_csv("tabella_test.csv")
tabella_train_cvs = pd.read_csv("tabella_train_set.csv")
tabella_test_cvs = pd.read_csv("tabella_test_set.csv")
tabella_train_cs_cv = pd.read_csv("Tabella_train_cross_subject_set.csv")
tabella_test_cs_cv = pd.read_csv("Tabella_test_cross_subject_set.csv")
tabella_train_cs = pd.read_csv("Tabella_train_cross_subject.csv")
tabella_test_cs = pd.read_csv("Tabella_test_cross_subject.csv")

def switch_train(train):
    if train == "tabella_train_ncs":
        return tabella_train_ncs

    elif train == "tabella_train_cvs":
        return tabella_train_cvs

    elif train == "tabella_train_cs_cv":
        return tabella_train_cs_cv

    elif train == "tabella_train_cs":
        return tabella_train_cs

tabella_train = switch_train("tabella_train_ncs")

def switch_test(test):
    if test == "tabella_test_ncs":
        return tabella_test_ncs
    elif test == "tabella_test_cvs":
        return tabella_test_cvs
    elif test == "tabella_test_cs_cv":
        return tabella_test_cs_cv
    elif test == "tabella_test_cs":
        return tabella_test_cs

tabella_test = switch_test("tabella_test_ncs")

tabella_train = tabella_train.iloc[:,1::]
tabella_test = tabella_test.iloc[:,1::]

y_train = []
for index,row in tabella_train.iterrows():
    if "A008" in row.iloc[0]:
        y = 0

    elif "A009" in row.iloc[0]:
        y = 1

    y_train.append(y)


y_train = np.array(y_train)

y_test = []
for index, row in tabella_test.iterrows():
    if "A008" in row.iloc[0]:
        y = 0

    elif "A009" in row.iloc[0]:
        y = 1

    y_test.append(y)

y_test = np.array(y_test)

x_train = tabella_train.iloc[:,1::].to_numpy() # converto la tabella in un array
x_test = tabella_test.iloc[:,1::].to_numpy()

class Addestra_modello_ML:
    def __init__(self, xtrain,ytrain,xtest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest

    def modello_KNN(self):
        parametri = [5, 'euclidean']
        knn_pipeline = Pipeline([('scaler', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=parametri[0],metric = parametri[1]))])
        #knn = KNeighborsClassifier(n_neighbors=parametri[0],metric = parametri[1])
        knn_pipeline.fit(self.xtrain,self.ytrain)
        predizioni = knn_pipeline.predict(self.xtest)

        return predizioni

    def modello_SVM(self):

        parametri = [0.5, "rbf"]
        svm_pipeline = Pipeline([('scaler',StandardScaler()),('svm', SVC(C=parametri[0], kernel=parametri[1]))])
        #svm = SVC(C=parametri[0], kernel=parametri[1])
        svm_pipeline.fit(self.xtrain,self.ytrain)
        predizioni = svm_pipeline.predict(self.xtest)

        return predizioni

    def modello_RF(self):

        parametri = [100,"gini",3,5,"sqrt"]
        rf_pipeline = Pipeline([('scaler',StandardScaler()),('rf', RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf = parametri[2],min_samples_split = parametri[3],max_features=parametri[4]))])
        #rf = RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf = parametri[2],min_samples_split = parametri[3],max_features=parametri[4])
        rf_pipeline.fit(self.xtrain,self.ytrain)
        predizioni = rf_pipeline.predict(self.xtest)

        return predizioni

    def modello_ADABOOST(self):

        parametri = [DecisionTreeClassifier(max_depth = 1),100,1]
        ab_pipeline = Pipeline([('scaler',StandardScaler()),('ab', AdaBoostClassifier(estimator=parametri[0],n_estimators=parametri[1],learning_rate=parametri[2], algorithm="SAMME"))])
        ab_pipeline.fit(self.xtrain, self.ytrain)
        predizioni = ab_pipeline.predict(self.xtest)

        return predizioni

    def modello_MLP(self):

        parametri = [(64,),0.001,500] # neuroni dello strato nascosto e learning rate init
        mlp_pipeline = Pipeline([('scaler',StandardScaler()),('mlp', MLPClassifier(hidden_layer_sizes=parametri[0],learning_rate_init=parametri[1],max_iter=parametri[2]))])
        mlp_pipeline.fit(self.xtrain,self.ytrain)
        predizioni = mlp_pipeline.predict(self.xtest)

        return predizioni

algoritmi_ml = Addestra_modello_ML(x_train,y_train,x_test)
predizioni_knn = algoritmi_ml.modello_KNN()
predizioni_svm = algoritmi_ml.modello_SVM()
predizioni_rf = algoritmi_ml.modello_RF()
predizioni_ab = algoritmi_ml.modello_ADABOOST()
predizioni_mlp = algoritmi_ml.modello_MLP()

# calcolo l'accuratezza

accuratezza_knn = accuracy_score(y_test, predizioni_knn)
accuratezza_rf = accuracy_score(y_test, predizioni_rf)
accuratezza_svm = accuracy_score(y_test, predizioni_svm)
accuratezza_ab = accuracy_score(y_test, predizioni_ab)
accuratezza_mlp = accuracy_score(y_test, predizioni_mlp)

print(f"accuratezza rf",accuratezza_rf)
print(f"accuratezza svm", accuratezza_svm)
print(f"accuratezza knn", accuratezza_knn)
print(f"accuratezza ab", accuratezza_ab)
print(f"accuratezza mlp", accuratezza_mlp)

# adesso faccio 100 simulazioni e ogni volta calcolo l'accuratezza

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

confusion_matrix_knn = np.zeros((2,2),dtype=int)
confusion_matrix_svm = np.zeros((2,2),dtype=int)
confusion_matrix_rf = np.zeros((2,2),dtype=int)
confusion_matrix_ab = np.zeros((2,2),dtype=int)
confusion_matrix_mlp = np.zeros((2,2),dtype=int)

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

for i in range(100):

    def switch_type(type_analisi,Tabella):
        if type_analisi == "analisi_cv":

            indici_vista_frontale2 = []
            for index, row in Tabella.iterrows():
                if "C002" in row.iloc[0]:
                    if "R002" in row.iloc[0]:
                        indici_vista_frontale2.append(index)


            Tabella = Tabella.iloc[indici_vista_frontale2, :]
            Tabella.reset_index(drop=True, inplace=True)  # resetto gli indici della tabella da 1 a n
            Tabella.index = Tabella.index + 1

            lista_nomi = []

            for index, row in Tabella.iterrows():

                lista_nomi.append(row.iloc[0])


            etichetta_tot_sub = []
            for j in range(len(indici_vista_frontale2)):
                etic = "FALSE"
                etichetta_tot_sub.append(etic)

            etichetta_tot_sett = []
            for j in range(len(indici_vista_frontale2)):
                etic = "TRUE"
                etichetta_tot_sett.append(etic)


            def estrai_sett(s):
                inizio = s.index("S")
                fine = inizio + 4
                return s[inizio:fine]

            setting = []
            for index, row in Tabella.iterrows():
                substring = estrai_sett(row.iloc[0])  # iloc vuole l'indice intero della posizione
                setting.append(substring)


            lista_set = sorted(set(setting))

            set_train = random.sample(lista_set, 12)  # restituisce casualmente gli elementi della lista senza ripetizioni
            set_test = []
            for sett in lista_set:
                if sett not in set_train:
                    set_test.append(sett)

            idx_train = []
            idx_test = []

            for sett in set_train:
                for index, row in Tabella.iterrows():
                    if sett in row.iloc[0]:
                        idx_train.append(index)

            for sett in set_test:
                for index, row in Tabella.iterrows():
                    if sett in row.iloc[0]:
                        idx_test.append(index)

            tabella_train = Tabella.loc[idx_train, :]
            tabella_test = Tabella.loc[idx_test, :]


            paz_train = tabella_train.iloc[:, 0]
            paz_test = tabella_test.iloc[:, 0]


            return tabella_train, tabella_test, etichetta_tot_sub, etichetta_tot_sett, paz_train, paz_test, lista_nomi

        elif type_analisi == "analisi_ncs":

            indici_vista_frontale2 = []
            for index, row in Tabella.iterrows():
                if "C002" in row.iloc[0]:
                    if "R002" in row.iloc[0]:
                        indici_vista_frontale2.append(index)

            Tabella = Tabella.iloc[indici_vista_frontale2, :]
            Tabella.reset_index(drop=True, inplace=True)  # resetto gli indici della tabella da 1 a n
            Tabella.index = Tabella.index + 1

            lista_nomi = []
            for index, row in Tabella.iterrows():

                lista_nomi.append(row.iloc[0])

            etichetta_tot_sub = []
            for j in range(len(indici_vista_frontale2)):

                etich = "FALSE"
                etichetta_tot_sub.append(etich)


            etichetta_tot_sett = []
            for j in range(len(indici_vista_frontale2)):
                etic = "FALSE"
                etichetta_tot_sett.append(etic)


            lista_index = []
            for index, row in Tabella.iterrows():
                lista_index.append(index)

            dim_train = round(70 * len(lista_index) / 100)
            idx_train = random.sample(lista_index, dim_train)

            idx_test = []
            for idx in lista_index:
                if idx not in idx_train:
                    idx_test.append(idx)

            tabella_train = Tabella.loc[idx_train, :]
            tabella_test = Tabella.loc[idx_test, :]


            paz_train = tabella_train.iloc[:, 0]
            paz_test = tabella_test.iloc[:, 0]


            return tabella_train, tabella_test, etichetta_tot_sub, etichetta_tot_sett, paz_train, paz_test, lista_nomi

        elif type_analisi == "analisi_cv_cs":

            indici_vista_frontale2 = []
            nomi_vista_frontale2 = []

            for index, row in Tabella.iterrows():
                if "C002" in row.iloc[0]:
                    if "R002" in row.iloc[0]:
                        indici_vista_frontale2.append(index)
                        nomi_vista_frontale2.append(row.iloc[0])

            Tabella = Tabella.iloc[indici_vista_frontale2, :]
            Tabella.reset_index(drop=True, inplace=True)
            Tabella.index = Tabella.index + 1

            lista_nomi = []
            for index, row in Tabella.iterrows():
                lista_nomi.append(row.iloc[0])


            # adesso faccio la suddivisione cross subject

            def estrai_paz(stringa):
                inizio = stringa.index('P')
                fine = inizio + 4

                return stringa[inizio:fine]

            lista_paz = []
            for index, row in Tabella.iterrows():
                paz = estrai_paz(row.iloc[0])
                lista_paz.append(paz)

            pazienti = sorted(set(lista_paz))  # il set toglie i duplicati

            paz_train = random.sample(pazienti, k=24)
            paz_test = []

            for paz in pazienti:
                if paz not in paz_train:
                    paz_test.append(paz)

            # faccio una prima suddivisione del train e test in base ai paz
            # qui ho riordinato i paz

            indici_train_cross_sub = []
            indici_test_cross_sub = []

            for paz in paz_train:
                for index, row in Tabella.iterrows():
                    if paz in row.iloc[0]:
                        indici_train_cross_sub.append(index)

            for paz in paz_test:
                for index, row in Tabella.iterrows():
                    if paz in row.iloc[0]:
                        indici_test_cross_sub.append(index)

            Tabella_train_cross_subject = Tabella.loc[indici_train_cross_sub, :]
            Tabella_test_cross_subject = Tabella.loc[indici_test_cross_sub, :]

            # adesso faccio la suddivisione cross setting

            def estrai_sett(stringa):
                inizio = stringa.index("S")
                fine = inizio + 4

                return stringa[inizio:fine]

            lista_set = []
            for index, row in Tabella.iterrows():
                setting = estrai_sett(row.iloc[0])
                lista_set.append(setting)

            setting = sorted(set(lista_set))
            set_train = random.sample(setting, k=10)
            set_test = []

            for sett in setting:
                if sett not in set_train:
                    set_test.append(sett)

            # qui ho riordinato i setting

            indici_train_cross_sub_set = []
            for sett in set_train:
                for index, row in Tabella_train_cross_subject.iterrows():
                    if sett in row.iloc[0]:
                        indici_train_cross_sub_set.append(index)

            indici_test_cross_sub_set = []
            for sett in set_test:
                for index, row in Tabella_test_cross_subject.iterrows():
                    if sett in row.iloc[0]:
                        indici_test_cross_sub_set.append(index)

            tabella_train = Tabella_train_cross_subject.loc[indici_train_cross_sub_set, :]
            tabella_test = Tabella_test_cross_subject.loc[indici_test_cross_sub_set, :]

            paz_train = tabella_train.iloc[:, 0]
            paz_test = tabella_test.iloc[:, 0]

            dim = len(paz_train)+len(paz_test)

            etichetta_tot_sub = []
            for j in range(dim):
                etich = "TRUE"
                etichetta_tot_sub.append(etich)

            etichetta_tot_sett = []
            for j in range(dim):
                etich = "TRUE"
                etichetta_tot_sett.append(etich)


            return tabella_train, tabella_test, etichetta_tot_sub, etichetta_tot_sett, paz_train, paz_test, lista_nomi

        elif type_analisi == "analisi_cs":

            indici_vista_frontale2 = []
            nomi_vista_frontale2 = []

            for index, row in Tabella.iterrows():
                if "C002" in row.iloc[0]:
                    if "R002" in row.iloc[0]:
                        indici_vista_frontale2.append(index)
                        nomi_vista_frontale2.append(row.iloc[0])

            Tabella = Tabella.iloc[indici_vista_frontale2, :]
            Tabella.reset_index(drop=True, inplace=True)
            Tabella.index = Tabella.index + 1

            lista_nomi = []
            for index, row in Tabella.iterrows():
                lista_nomi.append(row.iloc[0])


            etichetta_tot_sub = []
            for j in range(len(indici_vista_frontale2)):

                etic = "TRUE"
                etichetta_tot_sub.append(etic)


            etichetta_tot_sett = []
            for j in range(len(indici_vista_frontale2)):

                etic = "FALSE"
                etichetta_tot_sett.append(etic)


            # adesso faccio la suddivisione cross subject

            def estrai_paz(stringa):
                inizio = stringa.index('P')
                fine = inizio + 4

                return stringa[inizio:fine]

            lista_paz = []
            for index, row in Tabella.iterrows():
                paz = estrai_paz(row.iloc[0])
                lista_paz.append(paz)

            pazienti = sorted(set(lista_paz))  # il set toglie i duplicati

            paz_train = random.sample(pazienti, k=28)
            paz_test = []

            for paz in pazienti:
                if paz not in paz_train:
                    paz_test.append(paz)

            # faccio una prima suddivisione del train e test in base ai paz

            indici_train_cross_sub = []
            indici_test_cross_sub = []

            for paz in paz_train:
                for index, row in Tabella.iterrows():
                    if paz in row.iloc[0]:
                        indici_train_cross_sub.append(index)

            for paz in paz_test:
                for index, row in Tabella.iterrows():
                    if paz in row.iloc[0]:
                        indici_test_cross_sub.append(index)

            tabella_train = Tabella.loc[indici_train_cross_sub, :]
            tabella_test = Tabella.loc[indici_test_cross_sub, :]

            paz_train = tabella_train.iloc[:, 0]
            paz_test = tabella_test.iloc[:, 0]


            return tabella_train, tabella_test, etichetta_tot_sub, etichetta_tot_sett, paz_train, paz_test, lista_nomi


    [tabella_train, tabella_test, etichetta_tot_sub, etichetta_tot_sett, paz_train, paz_test, lista_nomi] = switch_type("analisi_cv_cs",Tabella)

    dim_train = len(tabella_train)
    dim_test = len(tabella_test)
    dim_tot = dim_train+dim_test

    vettore = np.full(dim_tot, dim_train)
    dim_train_sim.append(vettore)

    vettore = np.full(dim_tot , dim_test)
    dim_test_sim.append(vettore)


    vettore = np.full(dim_tot, i)
    id_sim.append(vettore)


    etichetta_is_sub_tot.append(etichetta_tot_sub)
    etichetta_is_sett_tot.append(etichetta_tot_sett)
    etichetta_is_tot_train.append(paz_train)
    etichetta_is_tot_test.append(paz_test)

    lista_nomi_sim.append(lista_nomi)


    y_train = []
    for index, row in tabella_train.iterrows():
        if "A008" in row.iloc[0]:
            y = 0

        elif "A009" in row.iloc[0]:
            y = 1

        y_train.append(y)

    y_train = np.array(y_train)

    y_test = []
    for index, row in tabella_test.iterrows():
        if "A008" in row.iloc[0]:
            y = 0

        elif "A009" in row.iloc[0]:
            y = 1

        y_test.append(y)

    y_test = np.array(y_test)


    x_train = tabella_train.iloc[:, 1::].to_numpy()  # converto la tabella in un array
    x_test = tabella_test.iloc[:, 1::].to_numpy()

    ytest_tot.append(y_test)
    ytrain_tot.append(y_train)


    class Addestramento_modello_ml:
        def __init__(self, xtrain, ytrain, xtest):
            self.xtrain = xtrain
            self.ytrain = ytrain
            self.xtest = xtest

        def modello_KNN(self):
            parametri = [5, 'euclidean']
            knn_pipeline = Pipeline([('scaler', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=parametri[0], metric=parametri[1]))])
            knn_pipeline.fit(self.xtrain, self.ytrain)
            predizioni_train = knn_pipeline.predict(self.xtrain)
            predizioni_test = knn_pipeline.predict(self.xtest)
            prob_train = knn_pipeline.predict_proba(self.xtrain)
            prob_test = knn_pipeline.predict_proba(self.xtest)

            return predizioni_test, predizioni_train, prob_train, prob_test, knn_pipeline

        def modello_SVM(self):
            parametri = [0.5, "rbf"]
            svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=parametri[0], kernel=parametri[1]))])
            svm_pipeline.fit(self.xtrain, self.ytrain)
            predizioni_train = svm_pipeline.predict(self.xtrain)
            predizioni_test = svm_pipeline.predict(self.xtest)

            decision_scores = svm_pipeline.decision_function(self.xtest)
            y_prob_test = expit(decision_scores)
            prob_test = np.stack([1 - y_prob_test, y_prob_test], -1)
            #prob_test = y_prob_test/np.sum(y_prob_test, axis = 1, keepdims = True)

            decision_scores = svm_pipeline.decision_function(self.xtrain)
            y_prob_train = expit(decision_scores)
            prob_train = np.stack([1 - y_prob_train, y_prob_train], -1)
            #prob_train = y_prob_train / np.sum(y_prob_train, axis=1, keepdims=True)

            return predizioni_test, predizioni_train,prob_train, prob_test, svm_pipeline

        def modello_RF(self):
            parametri = [100, "gini", 3, 5, "sqrt"]
            rf_pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=parametri[0],criterion=parametri[1],min_samples_leaf=parametri[2],min_samples_split=parametri[3],max_features=parametri[ 4]))])
            rf_pipeline.fit(self.xtrain, self.ytrain)
            predizioni_train = rf_pipeline.predict(self.xtrain)
            predizioni_test = rf_pipeline.predict(self.xtest)
            prob_train = rf_pipeline.predict_proba(self.xtrain)
            prob_test = rf_pipeline.predict_proba(self.xtest)

            return predizioni_test, predizioni_train, prob_train, prob_test, rf_pipeline

        def modello_ADABOOST(self):
            parametri = [DecisionTreeClassifier(max_depth=1), 200, 1]
            ab_pipeline = Pipeline([('scaler', StandardScaler()), ('ab', AdaBoostClassifier(estimator=parametri[0],n_estimators=parametri[1],learning_rate=parametri[2],algorithm="SAMME"))])
            ab_pipeline.fit(self.xtrain, self.ytrain)
            predizioni_train = ab_pipeline.predict(self.xtrain)
            predizioni_test = ab_pipeline.predict(self.xtest)
            prob_train = ab_pipeline.predict_proba(self.xtrain)
            prob_test = ab_pipeline.predict_proba(self.xtest)

            return predizioni_test, predizioni_train, prob_train, prob_test, ab_pipeline

        def modello_MLP(self):
            parametri = [(64,), 0.001,500]  # neuroni dello strato nascosto e learning rate init
            mlp_pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=parametri[0], activation = 'relu', solver = 'adam', learning_rate_init=parametri[1], max_iter=parametri[2]))])
            mlp_pipeline.fit(self.xtrain, self.ytrain)
            predizioni_train = mlp_pipeline.predict(self.xtrain)
            predizioni_test = mlp_pipeline.predict(self.xtest)
            prob_train = mlp_pipeline.predict_proba(self.xtrain)
            prob_test = mlp_pipeline.predict_proba(self.xtest)

            return predizioni_test, predizioni_train, prob_train, prob_test, mlp_pipeline

    algoritmi_ml = Addestramento_modello_ml(x_train, y_train, x_test)
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



    f1_knn_test = f1_score(y_test, predizioni_knn_test)
    f1_rf_test = f1_score(y_test, predizioni_rf_test)
    f1_svm_test = f1_score(y_test, predizioni_svm_test)
    f1_ab_test = f1_score(y_test, predizioni_ab_test)
    f1_mlp_test = f1_score(y_test, predizioni_mlp_test)

    lista_f1_knn_test.append(f1_knn_test)
    lista_f1_svm_test.append(f1_svm_test)
    lista_f1_rf_test.append(f1_rf_test)
    lista_f1_ab_test.append(f1_ab_test)
    lista_f1_mlp_test.append(f1_mlp_test)


    f1_knn_train = f1_score(y_train, predizioni_knn_train)
    f1_rf_train = f1_score(y_train, predizioni_rf_train)
    f1_svm_train = f1_score(y_train, predizioni_svm_train)
    f1_ab_train = f1_score(y_train, predizioni_ab_train)
    f1_mlp_train = f1_score(y_train, predizioni_mlp_train)


    lista_f1_knn_train.append(f1_knn_train)
    lista_f1_svm_train.append(f1_svm_train)
    lista_f1_rf_train.append(f1_rf_train)
    lista_f1_ab_train.append(f1_ab_train)
    lista_f1_mlp_train.append(f1_mlp_train)



    confusion_matrix_knn += confusion_matrix(y_test, predizioni_knn_test)
    confusion_matrix_svm += confusion_matrix(y_test, predizioni_svm_test)
    confusion_matrix_rf +=  confusion_matrix(y_test, predizioni_rf_test)
    confusion_matrix_ab +=  confusion_matrix(y_test, predizioni_ab_test)
    confusion_matrix_mlp += confusion_matrix(y_test, predizioni_mlp_test)




    print(f"iterazione",i)

print(lista_acc_knn_test)




confusion_matrix_knn = np.round(confusion_matrix_knn/100)
confusion_matrix_svm = np.round(confusion_matrix_svm/100)
confusion_matrix_rf = np.round(confusion_matrix_rf/100)
confusion_matrix_ab = np.round(confusion_matrix_ab/100)
confusion_matrix_mlp = np.round(confusion_matrix_mlp/100)




# salvataggio dati
cartella = "modelli"


with open(os.path.join(cartella,'predizioni_knn_cv_cs_test.pkl'),'wb') as file:
    pickle.dump(lista_predizioni_test_knn, file)

with open(os.path.join(cartella,'predizioni_svm_cv_cs_test.pkl'),'wb') as file:
    pickle.dump(lista_predizioni_test_svm, file)

with open(os.path.join(cartella,'predizioni_rf_cv_cs_test.pkl'),'wb') as file:
    pickle.dump(lista_predizioni_test_rf, file)

with open(os.path.join(cartella,'predizioni_ab_cv_cs_test.pkl'),'wb') as file:
    pickle.dump(lista_predizioni_test_ab, file)

with open(os.path.join(cartella,'predizioni_mlp_cv_cs_test.pkl'),'wb') as file:
    pickle.dump(lista_predizioni_test_mlp, file)

with open(os.path.join(cartella, 'predizioni_knn_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train_knn, file)

with open(os.path.join(cartella, 'predizioni_svm_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train_svm, file)

with open(os.path.join(cartella, 'predizioni_rf_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train_rf, file)

with open(os.path.join(cartella, 'predizioni_ab_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train_ab, file)

with open(os.path.join(cartella, 'predizioni_mlp_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_predizioni_train_mlp, file)

with open(os.path.join(cartella,'accuratezza_knn_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_knn_test,file)

with open(os.path.join(cartella,'accuratezza_svm_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_svm_test,file)

with open(os.path.join(cartella,'accuratezza_rf_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_rf_test,file)

with open(os.path.join(cartella,'accuratezza_ab_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_ab_test,file)

with open(os.path.join(cartella,'accuratezza_mlp_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_acc_mlp_test,file)

with open(os.path.join(cartella, 'accuratezza_knn_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_knn_train, file)

with open(os.path.join(cartella, 'accuratezza_svm_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_svm_train, file)

with open(os.path.join(cartella, 'accuratezza_rf_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_rf_train, file)

with open(os.path.join(cartella, 'accuratezza_ab_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_ab_train, file)

with open(os.path.join(cartella, 'accuratezza_mlp_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_acc_mlp_train, file)

with open(os.path.join(cartella, 'f1_knn_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_knn_test, file)

with open(os.path.join(cartella, 'f1_svm_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_svm_test, file)

with open(os.path.join(cartella, 'f1_rf_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_rf_test, file)

with open(os.path.join(cartella, 'f1_ab_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_ab_test, file)

with open(os.path.join(cartella, 'f1_mlp_cv_cs_test.pkl'), 'wb') as file:
    pickle.dump(lista_f1_mlp_test, file)

with open(os.path.join(cartella, 'f1_knn_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_knn_train, file)

with open(os.path.join(cartella, 'f1_svm_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_svm_train, file)

with open(os.path.join(cartella, 'f1_rf_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_rf_train, file)

with open(os.path.join(cartella, 'f1_ab_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_ab_train, file)

with open(os.path.join(cartella, 'f1_mlp_cv_cs_train.pkl'), 'wb') as file:
    pickle.dump(lista_f1_mlp_train, file)

with open(os.path.join(cartella,'simul_id_cv_cs.pkl'), 'wb') as file:
    pickle.dump(id_sim,file)

with open(os.path.join(cartella,'array_dim_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(dim_train_sim,file)

with open(os.path.join(cartella,'array_dim_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(dim_test_sim,file)

with open(os.path.join(cartella,'etichetta_is_sub_cv_cs.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_sub_tot, file)

with open(os.path.join(cartella,'etichetta_is_sett_cv_cs.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_sett_tot, file)

with open(os.path.join(cartella,'etichetta_is_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_train, file)

with open(os.path.join(cartella,'etichetta_is_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(etichetta_is_tot_test, file)

with open(os.path.join(cartella,'lista_nomi_simulazione_cv_cs.pkl'), 'wb') as file:
    pickle.dump(lista_nomi_sim,file)

with open(os.path.join(cartella,'is_pred_cc_knn_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_knn_train_tot,file)

with open(os.path.join(cartella,'is_pred_cc_svm_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_svm_train_tot,file)

with open(os.path.join(cartella,'is_pred_cc_rf_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_rf_train_tot,file)

with open(os.path.join(cartella,'is_pred_cc_ab_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_ab_train_tot,file)

with open(os.path.join(cartella,'is_pred_cc_mlp_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_mlp_train_tot,file)

with open(os.path.join(cartella,'is_pred_cc_knn_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_knn_test_tot,file)

with open(os.path.join(cartella,'is_pred_cc_svm_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_svm_test_tot,file)

with open(os.path.join(cartella,'is_pred_cc_rf_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_rf_test_tot,file)

with open(os.path.join(cartella,'is_pred_cc_ab_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_ab_test_tot,file)

with open(os.path.join(cartella,'is_pred_cc_mlp_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(is_pred_cc_mlp_test_tot,file)

with open(os.path.join(cartella,'item_loss_knn_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_knn_train_tot,file)

with open(os.path.join(cartella,'item_loss_svm_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_svm_train_tot,file)

with open(os.path.join(cartella,'item_loss_rf_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_rf_train_tot,file)

with open(os.path.join(cartella,'item_loss_ab_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_ab_train_tot,file)

with open(os.path.join(cartella,'item_loss_mlp_train_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_mlp_train_tot,file)

with open(os.path.join(cartella,'item_loss_knn_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_knn_test_tot,file)

with open(os.path.join(cartella,'item_loss_svm_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_svm_test_tot,file)

with open(os.path.join(cartella,'item_loss_rf_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_rf_test_tot,file)

with open(os.path.join(cartella,'item_loss_ab_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_ab_test_tot,file)

with open(os.path.join(cartella,'item_loss_mlp_test_cv_cs.pkl'), 'wb') as file:
    pickle.dump(item_loss_mlp_test_tot,file)

with open(os.path.join(cartella,'prob_conf_train_knn_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_knn_train_tot,file)

with open(os.path.join(cartella,'prob_conf_train_svm_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_svm_train_tot,file)

with open(os.path.join(cartella,'prob_conf_train_rf_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_rf_train_tot,file)

with open(os.path.join(cartella,'prob_conf_train_ab_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_ab_train_tot,file)

with open(os.path.join(cartella,'prob_conf_train_mlp_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_mlp_train_tot,file)

with open(os.path.join(cartella,'prob_conf_test_knn_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_knn_test_tot,file)

with open(os.path.join(cartella,'prob_conf_test_svm_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_svm_test_tot,file)

with open(os.path.join(cartella,'prob_conf_test_rf_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_rf_test_tot,file)

with open(os.path.join(cartella,'prob_conf_test_ab_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_ab_test_tot,file)

with open(os.path.join(cartella,'prob_conf_test_mlp_cv_cs.pkl'), 'wb') as file:
    pickle.dump(prob_mlp_test_tot,file)

with open(os.path.join(cartella,'ytest_cv_cs.pkl'), 'wb') as file:
    pickle.dump(ytest_tot,file)

with open(os.path.join(cartella,'ytrain_cv_cs.pkl'), 'wb') as file:
    pickle.dump(ytrain_tot,file)

with open(os.path.join(cartella,'confusion_matrix_knn_cv_cs.pkl'), 'wb') as file:
    pickle.dump(confusion_matrix_knn,file)

with open(os.path.join(cartella,'confusion_matrix_svm_cv_cs.pkl'), 'wb') as file:
    pickle.dump(confusion_matrix_svm,file)

with open(os.path.join(cartella,'confusion_matrix_rf_cv_cs.pkl'), 'wb') as file:
    pickle.dump(confusion_matrix_rf,file)

with open(os.path.join(cartella, 'confusion_matrix_ab_cv_cs.pkl'), 'wb') as file:
    pickle.dump(confusion_matrix_ab, file)

with open(os.path.join(cartella, 'confusion_matrix_mlp_cv_cs.pkl'), 'wb') as file:
    pickle.dump(confusion_matrix_mlp, file)


joblib.dump(modelli_knn, f"modelli/modelli_knn_cv_cs.pkl")
joblib.dump(modelli_svm, f"modelli/modelli_svm_cv_cs.pkl")
joblib.dump(modelli_rf, f"modelli/modelli_rf_cv_cs.pkl")
joblib.dump(modelli_ab, f"modelli/modelli_ab_cv_cs.pkl")
joblib.dump(modelli_mlp, f"modelli/modelli_mlp_cv_cs.pkl")
