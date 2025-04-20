import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import seaborn as sns


with open('accuratezza_modelli_cvs.pkl', 'rb') as file:
    accuratezza_modelli_cvs = pickle.load(file)

with open('accuratezza_modelli_ncs.pkl','rb') as file:
    accuratezza_modelli_ncs = pickle.load(file)

with open('accuratezza_modelli_cv_cs.pkl','rb') as file:
    accuratezza_modelli_cv_cs = pickle.load(file)

with open('accuratezza_modelli_cs.pkl','rb') as file:
    accuratezza_modelli_cs = pickle.load(file)

accuratezza_modelli_ncs = np.array(accuratezza_modelli_ncs)
accuratezza_modelli_ncs = accuratezza_modelli_ncs[0]
print(np.mean(accuratezza_modelli_ncs[0]),np.mean(accuratezza_modelli_ncs[1]),np.mean(accuratezza_modelli_ncs[2]),np.mean(accuratezza_modelli_ncs[3]),np.mean(accuratezza_modelli_ncs[4]))

accuratezza_modelli_cvs = np.array(accuratezza_modelli_cvs)
accuratezza_modelli_cvs = accuratezza_modelli_cvs[0]
print(np.mean(accuratezza_modelli_cvs[0]),np.mean(accuratezza_modelli_cvs[1]),np.mean(accuratezza_modelli_cvs[2]),np.mean(accuratezza_modelli_cvs[3]),np.mean(accuratezza_modelli_cvs[4]))

accuratezza_modelli_cv_cs = np.array(accuratezza_modelli_cv_cs)
accuratezza_modelli_cv_cs = accuratezza_modelli_cv_cs[0]
print(np.mean(accuratezza_modelli_cv_cs[0]),np.mean(accuratezza_modelli_cv_cs[1]),np.mean(accuratezza_modelli_cv_cs[2]),np.mean(accuratezza_modelli_cv_cs[3]),np.mean(accuratezza_modelli_cv_cs[4]))


accuratezza_modelli_cs = np.array(accuratezza_modelli_cs)
accuratezza_modelli_cs = accuratezza_modelli_cs[0]
print(np.mean(accuratezza_modelli_cs[0]),np.mean(accuratezza_modelli_cs[1]),np.mean(accuratezza_modelli_cs[2]),np.mean(accuratezza_modelli_cs[3]),np.mean(accuratezza_modelli_cs[4]))

# valuto la normalità dei modelli

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(accuratezza_modelli_ncs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-NCS ACCURATEZZA')
axes[0].set_ylabel('KNN')

stats.probplot(accuratezza_modelli_ncs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(accuratezza_modelli_ncs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(accuratezza_modelli_ncs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(accuratezza_modelli_ncs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(accuratezza_modelli_cvs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-CV ACCURATEZZA')
axes[0].set_ylabel('KNN')

stats.probplot(accuratezza_modelli_cvs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(accuratezza_modelli_cvs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(accuratezza_modelli_cvs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(accuratezza_modelli_cvs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(accuratezza_modelli_cv_cs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-CV_CS ACCURATEZZA')
axes[0].set_ylabel('KNN')

stats.probplot(accuratezza_modelli_cv_cs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(accuratezza_modelli_cv_cs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(accuratezza_modelli_cv_cs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(accuratezza_modelli_cv_cs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(accuratezza_modelli_cs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-CS ACCURATEZZA')
axes[0].set_ylabel('KNN')

stats.probplot(accuratezza_modelli_cs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(accuratezza_modelli_cs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(accuratezza_modelli_cs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(accuratezza_modelli_cs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modelli_ncs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA NCS ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].hist(accuratezza_modelli_ncs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(accuratezza_modelli_ncs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(accuratezza_modelli_ncs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(accuratezza_modelli_ncs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modelli_cvs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA CV ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].hist(accuratezza_modelli_cvs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(accuratezza_modelli_cvs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(accuratezza_modelli_cvs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(accuratezza_modelli_cvs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modelli_cv_cs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA CV-CS ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].hist(accuratezza_modelli_cv_cs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(accuratezza_modelli_cv_cs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(accuratezza_modelli_cv_cs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(accuratezza_modelli_cv_cs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modelli_cs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA CS ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].hist(accuratezza_modelli_cs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(accuratezza_modelli_cs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(accuratezza_modelli_cs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(accuratezza_modelli_cs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()

# boxplot


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modelli_ncs[0])
axes[0].set_title('ISTOGRAMMA NCS ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].boxplot(accuratezza_modelli_ncs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(accuratezza_modelli_ncs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(accuratezza_modelli_ncs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(accuratezza_modelli_ncs[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modelli_cvs[0])
axes[0].set_title('ISTOGRAMMA CV ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].boxplot(accuratezza_modelli_cvs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(accuratezza_modelli_cvs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(accuratezza_modelli_cvs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(accuratezza_modelli_cvs[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modelli_cv_cs[0])
axes[0].set_title('ISTOGRAMMA CV-CS ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].boxplot(accuratezza_modelli_cv_cs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(accuratezza_modelli_cv_cs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(accuratezza_modelli_cv_cs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(accuratezza_modelli_cv_cs[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modelli_cs[0])
axes[0].set_title('ISTOGRAMMA CS ACCURATEZZA')
axes[0].set_ylabel('KNN')

axes[1].boxplot(accuratezza_modelli_cs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(accuratezza_modelli_cs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(accuratezza_modelli_cs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(accuratezza_modelli_cs[4])
axes[4].set_ylabel('MLP')

plt.show()


def plot_statistics(d1, d2,desired_stats, desired_stats_names,descr1,descr2, fig_size = (10,6)):
    df1 = pd.DataFrame(data = d1)
    df1 = pd.melt(df1,value_vars = desired_stats,var_name = "Metric type",value_name = "Value")
    df1["Training type"] = descr1

    df2 = pd.DataFrame(data = d2)
    df2 = pd.melt(df2,value_vars = desired_stats,var_name = "Metric type",value_name = "Value")
    df2["Training type"] = descr2

    # concatenazione del dataframe
    df = pd.concat([df1, df2])

    # ridenominazione delle metriche

    df["Metric type"] = np.where(df["Metric type"].eq(desired_stats), desired_stats_names, df["Metric type"])

    # Creazione boxplot

    plt.figure(figsize = fig_size)
    sns.boxplot(x = "Metric type", y = "Value", data = df, hue = "Training type", palette = "Greens")
    plt.ylim(None)
    plt.title(f"Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "Accuratezza"
desired_stats_names = "Accuratezza (%)"

modello = ["knn","svm","rf","ab","mlp"]
for i in range(5):
    d1 = {"Accuratezza":accuratezza_modelli_ncs[i]}
    d2 = {"Accuratezza":accuratezza_modelli_cvs[i]}
    descr1 = "Modello NCV"
    descr2 = "Modello CV"

    plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


for i in range(5):
    d1 = {"Accuratezza":accuratezza_modelli_ncs[i]}
    d2 = {"Accuratezza":accuratezza_modelli_cs[i]}
    descr1 = "Modello NCV"
    descr2 = "Modello NCV-CS"

    plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


for i in range(5):
    d1 = {"Accuratezza":accuratezza_modelli_cvs[i]}
    d2 = {"Accuratezza":accuratezza_modelli_cv_cs[i]}
    descr1 = "Modello CV"
    descr2 = "Modello CS_CV"

    plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


# eseguo il test kolmogorov-smirnov per verificare la normalità

modelli = ['knn','svm','rf','ab','mlp']

for i in range(5):
    [dist_norm, p_value] = stats.kstest(accuratezza_modelli_ncs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.kstest(accuratezza_modelli_cvs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.kstest(accuratezza_modelli_cv_cs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.kstest(accuratezza_modelli_cs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)


for i in range(5):
    [dist_norm, p_value] = stats.shapiro(accuratezza_modelli_ncs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.shapiro(accuratezza_modelli_cvs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.shapiro(accuratezza_modelli_cv_cs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.shapiro(accuratezza_modelli_cs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

# i campioni dei modelli non seguono la distribuzione normale, ma popolazione numerosa

# adesso faccio il confronto statistico tra i vari modelli

# test per campioni indipendenti (Mann-Whitney U)

alternative = ['greater','less','two-sided']
alpha = 0.05

for alt in alternative:
    [stats_knn, p_value_knn] = stats.mannwhitneyu(accuratezza_modelli_ncs[0],accuratezza_modelli_cvs[0],alternative=alt)
    [stats_svm, p_value_svm] = stats.mannwhitneyu(accuratezza_modelli_ncs[1],accuratezza_modelli_cvs[1],alternative=alt)
    [stats_rf, p_value_rf] = stats.mannwhitneyu(accuratezza_modelli_ncs[2],accuratezza_modelli_cvs[2],alternative=alt)
    [stats_ab, p_value_ab] = stats.mannwhitneyu(accuratezza_modelli_ncs[3],accuratezza_modelli_cvs[3],alternative=alt)
    [stats_mlp, p_value_mlp] = stats.mannwhitneyu(accuratezza_modelli_ncs[4],accuratezza_modelli_cvs[4],alternative=alt)

    lista_p_value_acc = []

    lista_p_value_acc.append(p_value_knn)
    lista_p_value_acc.append(p_value_svm)
    lista_p_value_acc.append(p_value_rf)
    lista_p_value_acc.append(p_value_ab)
    lista_p_value_acc.append(p_value_mlp)


    for i in range(len(lista_p_value_acc)):
        if lista_p_value_acc[i] < alpha:
            print(f"Test Mann-Whitney, rifiuto H0 {modelli[i]}",lista_p_value_acc[i],alt)
        else:
            print(f"Test Mann-Whitney, accetto H0{modelli[i]}",lista_p_value_acc[i],alt)



for alt in alternative:
    [stats_knn, p_value_knn] = stats.mannwhitneyu(accuratezza_modelli_cvs[0],accuratezza_modelli_cv_cs[0],alternative=alt)
    [stats_svm, p_value_svm] = stats.mannwhitneyu(accuratezza_modelli_cvs[1],accuratezza_modelli_cv_cs[1],alternative=alt)
    [stats_rf, p_value_rf] = stats.mannwhitneyu(accuratezza_modelli_cvs[2],accuratezza_modelli_cv_cs[2],alternative=alt)
    [stats_ab, p_value_ab] = stats.mannwhitneyu(accuratezza_modelli_cvs[3],accuratezza_modelli_cv_cs[3],alternative=alt)
    [stats_mlp, p_value_mlp] = stats.mannwhitneyu(accuratezza_modelli_cvs[4],accuratezza_modelli_cv_cs[4],alternative=alt)

    lista_p_value_acc = []

    lista_p_value_acc.append(p_value_knn)
    lista_p_value_acc.append(p_value_svm)
    lista_p_value_acc.append(p_value_rf)
    lista_p_value_acc.append(p_value_ab)
    lista_p_value_acc.append(p_value_mlp)


    for i in range(len(lista_p_value_acc)):
        if lista_p_value_acc[i] < alpha:
            print(f"Test Mann-Whitney, rifiuto H0 {modelli[i]}",lista_p_value_acc[i],alt)
        else:
            print(f"Test Mann-Whitney, accetto H0 {modelli[i]}",lista_p_value_acc[i],alt)


for alt in alternative:
    [stats_knn, p_value_knn] = stats.mannwhitneyu(accuratezza_modelli_ncs[0],accuratezza_modelli_cs[0],alternative=alt)
    [stats_svm, p_value_svm] = stats.mannwhitneyu(accuratezza_modelli_ncs[1],accuratezza_modelli_cs[1],alternative=alt)
    [stats_rf, p_value_rf] = stats.mannwhitneyu(accuratezza_modelli_ncs[2],accuratezza_modelli_cs[2],alternative=alt)
    [stats_ab, p_value_ab] = stats.mannwhitneyu(accuratezza_modelli_ncs[3],accuratezza_modelli_cs[3],alternative=alt)
    [stats_mlp, p_value_mlp] = stats.mannwhitneyu(accuratezza_modelli_ncs[4],accuratezza_modelli_cs[4],alternative=alt)

    lista_p_value_acc = []

    lista_p_value_acc.append(p_value_knn)
    lista_p_value_acc.append(p_value_svm)
    lista_p_value_acc.append(p_value_rf)
    lista_p_value_acc.append(p_value_ab)
    lista_p_value_acc.append(p_value_mlp)


    for i in range(len(lista_p_value_acc)):
        if lista_p_value_acc[i] < alpha:
            print(f"Test Mann-Whitney, rifiuto H0 {modelli[i]}",lista_p_value_acc[i],alt)
        else:
            print(f"Test Mann-Whitney, accetto H0 {modelli[i]}",lista_p_value_acc[i],alt)


# adesso uso il t test per campioni diepndenti nell'ipotesi di normalità soddisfatta


# provo a supporre la normalità
# applico levene per verificare l'uguaglianza delle varianze

[stats_knn, p_value_knn] = stats.levene(accuratezza_modelli_ncs[0],accuratezza_modelli_cvs[0])
[stats_svm, p_value_svm] = stats.levene(accuratezza_modelli_ncs[1],accuratezza_modelli_cvs[1])
[stats_rf, p_value_rf] = stats.levene(accuratezza_modelli_ncs[2],accuratezza_modelli_cvs[2])
[stats_ab, p_value_ab] = stats.levene(accuratezza_modelli_ncs[3],accuratezza_modelli_cvs[3])
[stats_mlp, p_value_mlp] = stats.levene(accuratezza_modelli_ncs[4],accuratezza_modelli_cvs[4])

lista_p_value = []

lista_p_value.append(p_value_knn)
lista_p_value.append(p_value_svm)
lista_p_value.append(p_value_rf)
lista_p_value.append(p_value_ab)
lista_p_value.append(p_value_mlp)

alpha = 0.05
for p_value in lista_p_value:
    if p_value < alpha:
        print(f"Test Levene, rifiuto H0",p_value)
    else:
        print(f"Test Levene, accetto H0",p_value)

# adesso applico il ttest per campioni non appaiati


[stats_knn, p_value_knn] = stats.ttest_ind(accuratezza_modelli_ncs[0],accuratezza_modelli_cvs[0])
[stats_svm, p_value_svm] = stats.ttest_ind(accuratezza_modelli_ncs[1],accuratezza_modelli_cvs[1])
[stats_rf, p_value_rf] = stats.ttest_ind(accuratezza_modelli_ncs[2],accuratezza_modelli_cvs[2])
[stats_ab, p_value_ab] = stats.ttest_ind(accuratezza_modelli_ncs[3],accuratezza_modelli_cvs[3])
[stats_mlp, p_value_mlp] = stats.ttest_ind(accuratezza_modelli_ncs[4],accuratezza_modelli_cvs[4])

lista_p_value = []

lista_p_value.append(p_value_knn)
lista_p_value.append(p_value_svm)
lista_p_value.append(p_value_rf)
lista_p_value.append(p_value_ab)
lista_p_value.append(p_value_mlp)

alpha = 0.05
for p_value in lista_p_value:
    if p_value < alpha:
        print(f"Test T, rifiuto H0",p_value)
    else:
        print(f"Test T, accetto H0",p_value)




[stats_knn, p_value_knn] = stats.levene(accuratezza_modelli_ncs[0],accuratezza_modelli_cs[0])

alpha = 0.05
if p_value_knn < alpha:
    print(f"Test Levene, rifiuto H0",p_value_knn)
else:
    print(f"Test Levene, accetto H0",p_value_knn)

# adesso applico il ttest per campioni non appaiati


for alt in alternative:
    [stats_knn, p_value_knn] = stats.ttest_ind(accuratezza_modelli_ncs[0],accuratezza_modelli_cs[0],alternative = alt)

    if p_value_knn < alpha:
        print(f"Test T ncs-cs, rifiuto H0",p_value_knn)
    else:
        print(f"Test T ncs-cs, accetto H0",p_value_knn)



# adesso calcolo gli intervalli di confidenza
# ho n > 30

def intervalli_confidenza_acc(accuratezza):
    alpha = 0.05
    index_low = (alpha/2)*100
    index_high = (1-alpha/2)*100
    acc_ordin = np.sort(accuratezza)
    pmin = acc_ordin[round(index_low)]
    pmax = acc_ordin[round(index_high)]

    return pmin, pmax

int_conf_ncs = []
lista_int_conf_ncs_acc = []
lista_marg_errore_acc_ncs = []
lista_err_upper_ncs_acc = []
lista_err_lower_ncs_acc = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_acc(accuratezza_modelli_ncs[i])
    err_lower = np.mean(accuratezza_modelli_ncs[i]) - p_min
    err_upper = p_max - np.mean(accuratezza_modelli_ncs[i])
    lista_err_upper_ncs_acc.append(err_upper)
    lista_err_lower_ncs_acc.append(err_lower)
    int_conf_ncs.append((p_min,p_max))

for i in range(5):
    value = list(int_conf_ncs[i])
    lista_int_conf_ncs_acc.append(value)


int_conf_cvs = []
lista_int_conf_cvs_acc = []
lista_marg_errore_acc_cvs = []
lista_err_upper_cvs_acc = []
lista_err_lower_cvs_acc = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_acc(accuratezza_modelli_cvs[i])
    err_lower = np.mean(accuratezza_modelli_cvs[i]) - p_min
    err_upper = p_max - np.mean(accuratezza_modelli_cvs[i])
    lista_err_upper_cvs_acc.append(err_upper)
    lista_err_lower_cvs_acc.append(err_lower)
    int_conf_cvs.append((p_min,p_max))

for i in range(5):
    value = list(int_conf_cvs[i])
    lista_int_conf_cvs_acc.append(value)

int_conf_cv_cs = []
lista_int_conf_cv_cs_acc = []
lista_marg_errore_acc_cv_cs = []
lista_err_upper_cv_cs_acc = []
lista_err_lower_cv_cs_acc = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_acc(accuratezza_modelli_cv_cs[i])
    err_lower = np.mean(accuratezza_modelli_cv_cs[i]) - p_min
    err_upper = p_max - np.mean(accuratezza_modelli_cv_cs[i])
    lista_err_upper_cv_cs_acc.append(err_upper)
    lista_err_lower_cv_cs_acc.append(err_lower)
    int_conf_cv_cs.append((p_min,p_max))

for i in range(5):
    value = list(int_conf_cv_cs[i])
    lista_int_conf_cv_cs_acc.append(value)

int_conf_cs = []
lista_int_conf_cs_acc = []
lista_marg_errore_acc_cs = []
lista_err_upper_cs_acc = []
lista_err_lower_cs_acc = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_acc(accuratezza_modelli_cs[i])
    err_lower = np.mean(accuratezza_modelli_cs[i]) - p_min
    err_upper = p_max - np.mean(accuratezza_modelli_cs[i])
    lista_err_upper_cs_acc.append(err_upper)
    lista_err_lower_cs_acc.append(err_lower)
    int_conf_cs.append((p_min,p_max))

for i in range(5):
    value = list(int_conf_cs[i])
    lista_int_conf_cs_acc.append(value)

print('inter cvs',lista_int_conf_cvs_acc)
print('inter ncs',lista_int_conf_ncs_acc)
print('inter cv_cs',lista_int_conf_cv_cs_acc)
print('inter cs',lista_int_conf_cs_acc)



with open('f1_modelli_cvs.pkl', 'rb') as file:
    f1_modelli_cvs = pickle.load(file)

with open('f1_modelli_ncs.pkl','rb') as file:
    f1_modelli_ncs = pickle.load(file)

with open('f1_modelli_cv_cs.pkl', 'rb') as file:
    f1_modelli_cv_cs = pickle.load(file)

with open('f1_modelli_cs.pkl', 'rb') as file:
    f1_modelli_cs = pickle.load(file)

f1_modelli_ncs = np.array(f1_modelli_ncs)
f1_modelli_ncs = f1_modelli_ncs[0]
print(np.mean(f1_modelli_ncs[0]),np.mean(f1_modelli_ncs[1]),np.mean(f1_modelli_ncs[2]),np.mean(f1_modelli_ncs[3]),np.mean(f1_modelli_ncs[4]))

f1_modelli_cvs = np.array(f1_modelli_cvs)
f1_modelli_cvs = f1_modelli_cvs[0]
print(np.mean(f1_modelli_cvs[0]),np.mean(f1_modelli_cvs[1]),np.mean(f1_modelli_cvs[2]),np.mean(f1_modelli_cvs[3]),np.mean(f1_modelli_cvs[4]))

f1_modelli_cv_cs = np.array(f1_modelli_cv_cs)
f1_modelli_cv_cs = f1_modelli_cv_cs[0]
print(np.mean(f1_modelli_cv_cs[0]),np.mean(f1_modelli_cv_cs[1]),np.mean(f1_modelli_cv_cs[2]),np.mean(f1_modelli_cv_cs[3]),np.mean(f1_modelli_cv_cs[4]))

f1_modelli_cs = np.array(f1_modelli_cs)
f1_modelli_cs = f1_modelli_cs[0]
print(np.mean(f1_modelli_cs[0]),np.mean(f1_modelli_cs[1]),np.mean(f1_modelli_cs[2]),np.mean(f1_modelli_cs[3]),np.mean(f1_modelli_cs[4]))

# valuto la normalità dei modelli

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(f1_modelli_ncs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-NCS F1')
axes[0].set_ylabel('KNN')

stats.probplot(f1_modelli_ncs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(f1_modelli_ncs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(f1_modelli_ncs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(f1_modelli_ncs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(f1_modelli_cvs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-CV F1')
axes[0].set_ylabel('KNN')

stats.probplot(f1_modelli_cvs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(f1_modelli_cvs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(f1_modelli_cvs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(f1_modelli_cvs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(f1_modelli_cv_cs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-CV-CS F1')
axes[0].set_ylabel('KNN')

stats.probplot(f1_modelli_cv_cs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(f1_modelli_cv_cs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(f1_modelli_cv_cs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(f1_modelli_cv_cs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')



fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

stats.probplot(f1_modelli_cs[0], dist='norm', plot=axes[0])
axes[0].set_title('QQ-plot-CS F1')
axes[0].set_ylabel('KNN')

stats.probplot(f1_modelli_cs[1], dist='norm', plot=axes[1])
axes[1].set_ylabel('SVM')

stats.probplot(f1_modelli_cs[2], dist='norm', plot=axes[2])
axes[2].set_ylabel('RF')

stats.probplot(f1_modelli_cs[3], dist='norm', plot=axes[3])
axes[3].set_ylabel('AB')

stats.probplot(f1_modelli_cs[4], dist='norm', plot=axes[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(f1_modelli_ncs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA NCS F1')
axes[0].set_ylabel('KNN')

axes[1].hist(f1_modelli_ncs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(f1_modelli_ncs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(f1_modelli_ncs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(f1_modelli_ncs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()

fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(f1_modelli_cvs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA CV F1')
axes[0].set_ylabel('KNN')

axes[1].hist(f1_modelli_cvs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(f1_modelli_cvs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(f1_modelli_cvs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(f1_modelli_cvs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()



fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(f1_modelli_cv_cs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA CV-CS F1')
axes[0].set_ylabel('KNN')

axes[1].hist(f1_modelli_cv_cs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(f1_modelli_cv_cs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(f1_modelli_cv_cs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(f1_modelli_cv_cs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()



fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].hist(f1_modelli_cv_cs[0], bins = 10)
axes[0].set_title('ISTOGRAMMA CS F1')
axes[0].set_ylabel('KNN')

axes[1].hist(f1_modelli_cs[1], bins = 10)
axes[1].set_ylabel('SVM')

axes[2].hist(f1_modelli_cs[2], bins = 10)
axes[2].set_ylabel('RF')

axes[3].hist(f1_modelli_cs[3], bins = 10)
axes[3].set_ylabel('AB')

axes[4].hist(f1_modelli_cs[4], bins = 10)
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(f1_modelli_ncs[0])
axes[0].set_title('ISTOGRAMMA NCS F1')
axes[0].set_ylabel('KNN')

axes[1].boxplot(f1_modelli_ncs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(f1_modelli_ncs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(f1_modelli_ncs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(f1_modelli_ncs[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(f1_modelli_cvs[0])
axes[0].set_title('ISTOGRAMMA CV F1')
axes[0].set_ylabel('KNN')

axes[1].boxplot(f1_modelli_cvs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(f1_modelli_cvs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(f1_modelli_cvs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(f1_modelli_cvs[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(f1_modelli_cv_cs[0])
axes[0].set_title('BOXPLOT CV-CS F1')
axes[0].set_ylabel('KNN')

axes[1].boxplot(f1_modelli_cv_cs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(f1_modelli_cv_cs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(f1_modelli_cv_cs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(f1_modelli_cv_cs[4])
axes[4].set_ylabel('MLP')

plt.show()


fig, axes = plt.subplots(nrows = 5,ncols = 1, sharex = True)

axes[0].boxplot(f1_modelli_cs[0])
axes[0].set_title('BOX-PLOT CS F1')
axes[0].set_ylabel('KNN')

axes[1].boxplot(f1_modelli_cs[1])
axes[1].set_ylabel('SVM')

axes[2].boxplot(f1_modelli_cs[2])
axes[2].set_ylabel('RF')

axes[3].boxplot(f1_modelli_cs[3])
axes[3].set_ylabel('AB')

axes[4].boxplot(f1_modelli_cs[4])
axes[4].set_ylabel('MLP')

plt.show()


def plot_statistics(d1, d2,desired_stats, desired_stats_names,descr1,descr2, fig_size = (10,6)):
    df1 = pd.DataFrame(data = d1)
    df1 = pd.melt(df1,value_vars = desired_stats,var_name = "Metric type",value_name = "Value")
    df1["Training type"] = descr1

    df2 = pd.DataFrame(data = d2)
    df2 = pd.melt(df2,value_vars = desired_stats,var_name = "Metric type",value_name = "Value")
    df2["Training type"] = descr2

    # concatenazione del dataframe
    df = pd.concat([df1, df2])

    # ridenominazione delle metriche

    df["Metric type"] = np.where(df["Metric type"].eq(desired_stats), desired_stats_names, df["Metric type"])

    # Creazione boxplot

    plt.figure(figsize = fig_size)
    sns.boxplot(x = "Metric type", y = "Value", data = df, hue = "Training type", palette = "Greens")
    plt.ylim(None)
    plt.title(f"Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "F1_Score"
desired_stats_names = "F1_Score (%)"

modello = ["knn","svm","rf","ab","mlp"]
for i in range(5):
    d1 = {"F1_Score":f1_modelli_ncs[i]}
    d2 = {"F1_Score":f1_modelli_cvs[i]}
    descr1 = "Modello NCV"
    descr2 = "Modello CV"

    plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


for i in range(5):
    d1 = {"F1_Score":f1_modelli_ncs[i]}
    d2 = {"F1_Score":f1_modelli_cs[i]}
    descr1 = "Modello NCV"
    descr2 = "Modello NCV-CS"

    plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


for i in range(5):
    d1 = {"F1_Score":f1_modelli_cvs[i]}
    d2 = {"F1_Score":f1_modelli_cv_cs[i]}
    descr1 = "Modello CV"
    descr2 = "Modello CS_CV"

    plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


# eseguo il test kolmogorov-smirnov per verificare la normalità

modelli = ['knn','svm','rf','ab','mlp']

for i in range(5):
    [dist_norm, p_value] = stats.kstest(f1_modelli_ncs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.kstest(f1_modelli_cvs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)


for i in range(5):
    [dist_norm, p_value] = stats.kstest(f1_modelli_cv_cs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.kstest(f1_modelli_cs[i], 'norm')
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)


for i in range(5):
    [dist_norm, p_value] = stats.shapiro(f1_modelli_ncs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.shapiro(f1_modelli_cvs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.shapiro(f1_modelli_cv_cs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}",p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}",p_value)

for i in range(5):
    [dist_norm, p_value] = stats.shapiro(f1_modelli_cs[i])
    if p_value < 0.05:
        print(f"rifiuto l'ipotesi nulla per {modelli[i]}", p_value)
    elif p_value > 0.05:
        print(f"accetto l'ipotesi per {modelli[i]}", p_value)

# i campioni dei modelli non seguono la distribuzione normale, ma popolazione numerosa

# adesso faccio il confronto statistico tra i vari modelli

# test per campioni indipendenti (Mann-Whitney U)
alpha = 0.05
for alt in alternative:
    [stats_knn, p_value_knn] = stats.mannwhitneyu(f1_modelli_ncs[0],f1_modelli_cvs[0],alternative=alt)
    [stats_svm, p_value_svm] = stats.mannwhitneyu(f1_modelli_ncs[1],f1_modelli_cvs[1],alternative=alt)
    [stats_rf, p_value_rf] = stats.mannwhitneyu(f1_modelli_ncs[2],f1_modelli_cvs[2],alternative=alt)
    [stats_ab, p_value_ab] = stats.mannwhitneyu(f1_modelli_ncs[3],f1_modelli_cvs[3],alternative=alt)
    [stats_mlp, p_value_mlp] = stats.mannwhitneyu(f1_modelli_ncs[4],f1_modelli_cvs[4],alternative=alt)

    lista_p_value_f1 = []

    lista_p_value_f1.append(p_value_knn)
    lista_p_value_f1.append(p_value_svm)
    lista_p_value_f1.append(p_value_rf)
    lista_p_value_f1.append(p_value_ab)
    lista_p_value_f1.append(p_value_mlp)

    for i in range(len(lista_p_value_f1)):
        if lista_p_value_f1[i] < alpha:
            print(f"Test Mann-Whitney, rifiuto H0 {modelli[i]}",lista_p_value_f1[i],alt)
        else:
            print(f"Test Mann-Whitney, accetto H0 {modelli[i]}",lista_p_value_f1[i],alt)


for alt in alternative:
    [stats_knn, p_value_knn] = stats.mannwhitneyu(f1_modelli_cvs[0],f1_modelli_cv_cs[0],alternative=alt)
    [stats_svm, p_value_svm] = stats.mannwhitneyu(f1_modelli_cvs[1],f1_modelli_cv_cs[1],alternative=alt)
    [stats_rf, p_value_rf] = stats.mannwhitneyu(f1_modelli_cvs[2],f1_modelli_cv_cs[2],alternative=alt)
    [stats_ab, p_value_ab] = stats.mannwhitneyu(f1_modelli_cvs[3],f1_modelli_cv_cs[3],alternative=alt)
    [stats_mlp, p_value_mlp] = stats.mannwhitneyu(f1_modelli_cvs[4],f1_modelli_cv_cs[4],alternative=alt)

    lista_p_value_f1 = []

    lista_p_value_f1.append(p_value_knn)
    lista_p_value_f1.append(p_value_svm)
    lista_p_value_f1.append(p_value_rf)
    lista_p_value_f1.append(p_value_ab)
    lista_p_value_f1.append(p_value_mlp)

    for i in range(len(lista_p_value_f1)):
        if lista_p_value_f1[i] < alpha:
            print(f"Test Mann-Whitney, rifiuto H0 {modelli[i]}",lista_p_value_f1[i],alt)
        else:
            print(f"Test Mann-Whitney, accetto H0{modelli[i]}",lista_p_value_f1[i],alt)


for alt in alternative:
    [stats_knn, p_value_knn] = stats.mannwhitneyu(f1_modelli_ncs[0],f1_modelli_cs[0],alternative=alt)
    [stats_svm, p_value_svm] = stats.mannwhitneyu(f1_modelli_ncs[1],f1_modelli_cs[1],alternative=alt)
    [stats_rf, p_value_rf] = stats.mannwhitneyu(f1_modelli_ncs[2],f1_modelli_cs[2],alternative=alt)
    [stats_ab, p_value_ab] = stats.mannwhitneyu(f1_modelli_ncs[3],f1_modelli_cs[3],alternative=alt)
    [stats_mlp, p_value_mlp] = stats.mannwhitneyu(f1_modelli_ncs[4],f1_modelli_cs[4],alternative=alt)

    lista_p_value_f1 = []

    lista_p_value_f1.append(p_value_knn)
    lista_p_value_f1.append(p_value_svm)
    lista_p_value_f1.append(p_value_rf)
    lista_p_value_f1.append(p_value_ab)
    lista_p_value_f1.append(p_value_mlp)

    for i in range(len(lista_p_value_f1)):
        if lista_p_value_f1[i] < alpha:
            print(f"Test Mann-Whitney, rifiuto H0 {modelli[i]}",lista_p_value_f1[i],alt)
        else:
            print(f"Test Mann-Whitney, accetto H0 {modelli[i]}",lista_p_value_f1[i],alt)



# adesso uso il t test per campioni diepndenti nell'ipotesi di normalità soddisfatta


# provo a supporre la normalità
# applico levene per verificare l'uguaglianza delle varianze

[stats_knn, p_value_knn] = stats.levene(f1_modelli_ncs[0],f1_modelli_cvs[0])
[stats_svm, p_value_svm] = stats.levene(f1_modelli_ncs[1],f1_modelli_cvs[1])
[stats_rf, p_value_rf] = stats.levene(f1_modelli_ncs[2],f1_modelli_cvs[2])
[stats_ab, p_value_ab] = stats.levene(f1_modelli_ncs[3],f1_modelli_cvs[3])
[stats_mlp, p_value_mlp] = stats.levene(f1_modelli_ncs[4],f1_modelli_cvs[4])

lista_p_value = []

lista_p_value.append(p_value_knn)
lista_p_value.append(p_value_svm)
lista_p_value.append(p_value_rf)
lista_p_value.append(p_value_ab)
lista_p_value.append(p_value_mlp)

alpha = 0.05
for p_value in lista_p_value:
    if p_value < alpha:
        print(f"Test Levene, rifiuto H0",p_value)
    else:
        print(f"Test Levene, accetto H0",p_value)

# adesso applico il ttest per campioni non appaiati


[stats_knn, p_value_knn] = stats.ttest_ind(f1_modelli_ncs[0],f1_modelli_cvs[0])
[stats_svm, p_value_svm] = stats.ttest_ind(f1_modelli_ncs[1],f1_modelli_cvs[1])
[stats_rf, p_value_rf] = stats.ttest_ind(f1_modelli_ncs[2],f1_modelli_cvs[2])
[stats_ab, p_value_ab] = stats.ttest_ind(f1_modelli_ncs[3],f1_modelli_cvs[3])
[stats_mlp, p_value_mlp] = stats.ttest_ind(f1_modelli_ncs[4],f1_modelli_cvs[4])

lista_p_value = []

lista_p_value.append(p_value_knn)
lista_p_value.append(p_value_svm)
lista_p_value.append(p_value_rf)
lista_p_value.append(p_value_ab)
lista_p_value.append(p_value_mlp)

alpha = 0.05
for p_value in lista_p_value:
    if p_value < alpha:
        print(f"Test T, rifiuto H0",p_value)
    else:
        print(f"Test T, accetto H0",p_value)


[stats_knn, p_value_knn] = stats.levene(f1_modelli_ncs[0],f1_modelli_cs[0])

alpha = 0.05
if p_value_knn < alpha:
    print(f"Test Levene, rifiuto H0",p_value_knn)
else:
    print(f"Test Levene, accetto H0",p_value_knn)

# adesso applico il ttest per campioni non appaiati


for alt in alternative:
    [stats_knn, p_value_knn] = stats.ttest_ind(f1_modelli_ncs[0],f1_modelli_cs[0],alternative = alt)

    if p_value_knn < alpha:
        print(f"Test T ncs-cs, rifiuto H0",p_value_knn)
    else:
        print(f"Test T ncs-cs, accetto H0",p_value_knn)

# adesso calcolo gli intervalli di confidenza
# ho n > 30

def intervalli_confidenza_f1(f1):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    f1_ordin = np.sort(f1)
    pmin = f1_ordin[round(index_low)]
    pmax = f1_ordin[round(index_high)]

    return pmin, pmax

int_conf_ncs = []
lista_int_conf_ncs_f1 = []
lista_marg_errore_f1_ncs = []
lista_err_upper_ncs_f1 = []
lista_err_lower_ncs_f1 = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_f1(f1_modelli_ncs[i])
    err_lower = np.mean(f1_modelli_ncs[i]) - p_min
    err_upper = p_max - np.mean(f1_modelli_ncs[i])
    lista_err_upper_ncs_f1.append(err_upper)
    lista_err_lower_ncs_f1.append(err_lower)
    int_conf_ncs.append((p_min,p_max))

for i in range(5):
    value = list(int_conf_ncs[i])
    lista_int_conf_ncs_f1.append(value)

int_conf_cvs = []
lista_int_conf_cvs_f1 = []
lista_marg_errore_f1_cvs = []
lista_err_upper_cvs_f1 = []
lista_err_lower_cvs_f1 = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_f1(f1_modelli_cvs[i])
    err_lower = np.mean(f1_modelli_cvs[i]) - p_min
    err_upper = p_max - np.mean(f1_modelli_cvs[i])
    lista_err_upper_cvs_f1.append(err_upper)
    lista_err_lower_cvs_f1.append(err_lower)
    int_conf_cvs.append((p_min,p_max))

for i in range(5):
    value = list(int_conf_cvs[i])
    lista_int_conf_cvs_f1.append(value)

int_conf_cv_cs = []
lista_int_conf_cv_cs_f1 = []
lista_marg_errore_f1_cv_cs = []
lista_err_upper_cv_cs_f1 = []
lista_err_lower_cv_cs_f1 = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_f1(f1_modelli_cv_cs[i])
    err_lower = np.mean(f1_modelli_cv_cs[i]) - p_min
    err_upper = p_max - np.mean(f1_modelli_cv_cs[i])
    lista_err_upper_cv_cs_f1.append(err_upper)
    lista_err_lower_cv_cs_f1.append(err_lower)
    int_conf_cv_cs.append((p_min, p_max))

for i in range(5):
    value = list(int_conf_cv_cs[i])
    lista_int_conf_cv_cs_f1.append(value)

int_conf_cs = []
lista_int_conf_cs_f1 = []
lista_marg_errore_f1_cs = []
lista_err_upper_cs_f1 = []
lista_err_lower_cs_f1 = []
for i in range(5):
    [p_min, p_max] = intervalli_confidenza_f1(f1_modelli_cs[i])
    err_lower = np.mean(f1_modelli_cs[i]) - p_min
    err_upper = p_max - np.mean(f1_modelli_cs[i])
    lista_err_upper_cs_f1.append(err_upper)
    lista_err_lower_cs_f1.append(err_lower)
    int_conf_cs.append((p_min, p_max))

for i in range(5):
    value = list(int_conf_cs[i])
    lista_int_conf_cs_f1.append(value)


print('inter cvs',lista_int_conf_cvs_f1)
print('inter ncs',lista_int_conf_ncs_f1)
print('inter cv_cs',lista_int_conf_cv_cs_f1)
print('inter cs',lista_int_conf_cs_f1)


lista_err_acc_f1_cvs_lower = []
lista_err_acc_f1_cvs_upper = []
for i in range(5):
    value1 = lista_err_lower_cvs_acc[i]
    value2 = lista_err_lower_cvs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_cvs_lower.append(value)


for i in range(5):
    value1 = lista_err_upper_cvs_acc[i]
    value2 = lista_err_upper_cvs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_cvs_upper.append(value)


lista_err_acc_f1_ncs_lower = []
lista_err_acc_f1_ncs_upper = []
for i in range(5):
    value1 = lista_err_lower_ncs_acc[i]
    value2 = lista_err_lower_ncs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_ncs_lower.append(value)

for i in range(5):
    value1 = lista_err_upper_ncs_acc[i]
    value2 = lista_err_upper_ncs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_ncs_upper.append(value)



lista_err_acc_f1_cv_cs_lower = []
lista_err_acc_f1_cv_cs_upper = []
for i in range(5):
    value1 = lista_err_lower_cv_cs_acc[i]
    value2 = lista_err_lower_cv_cs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_cv_cs_lower.append(value)

for i in range(5):
    value1 = lista_err_upper_cv_cs_acc[i]
    value2 = lista_err_upper_cv_cs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_cv_cs_upper.append(value)


lista_err_acc_f1_cs_lower = []
lista_err_acc_f1_cs_upper = []
for i in range(5):
    value1 = lista_err_lower_cs_acc[i]
    value2 = lista_err_lower_cs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_cs_lower.append(value)

for i in range(5):
    value1 = lista_err_upper_cs_acc[i]
    value2 = lista_err_upper_cs_f1[i]
    value = [value1, value2]
    lista_err_acc_f1_cs_upper.append(value)

lista_mean_acc_f1_cvs = []
for i in range(5):
    value1 = np.mean(accuratezza_modelli_cvs[i])
    value2 = np.mean(f1_modelli_cvs[i])
    value = [value1, value2]
    lista_mean_acc_f1_cvs.append(value)

lista_mean_acc_f1_ncs = []
for i in range(5):
    value1 = np.mean(accuratezza_modelli_ncs[i])
    value2 = np.mean(f1_modelli_ncs[i])
    value = [value1, value2]
    lista_mean_acc_f1_ncs.append(value)

lista_mean_acc_f1_cv_cs = []
for i in range(5):
    value1 = np.mean(accuratezza_modelli_cv_cs[i])
    value2 = np.mean(f1_modelli_cv_cs[i])
    value = [value1, value2]
    lista_mean_acc_f1_cv_cs.append(value)

lista_mean_acc_f1_cs = []
for i in range(5):
    value1 = np.mean(accuratezza_modelli_cs[i])
    value2 = np.mean(f1_modelli_cs[i])
    value = [value1, value2]
    lista_mean_acc_f1_cs.append(value)


# grafico ncs vs cvs

yerr_cvs = [lista_err_acc_f1_cvs_lower[0], lista_err_acc_f1_cvs_upper[0]]
yerr_ncs = [lista_err_acc_f1_ncs_lower[0], lista_err_acc_f1_ncs_upper[0]]

metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica)) # ottengo un array ([0,1])
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[0], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cvs[0], width, yerr = yerr_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del KNN')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[1], lista_err_acc_f1_cvs_upper[1]]
yerr_ncs = [lista_err_acc_f1_ncs_lower[1], lista_err_acc_f1_ncs_upper[1]]


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[1], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cvs[1], width, yerr = yerr_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del SVM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[2], lista_err_acc_f1_cvs_upper[2]]
yerr_ncs = [lista_err_acc_f1_ncs_lower[2], lista_err_acc_f1_ncs_upper[2]]

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[2], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cvs[2], width, yerr = yerr_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del RF')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[3], lista_err_acc_f1_cvs_upper[3]]
yerr_ncs = [lista_err_acc_f1_ncs_lower[3], lista_err_acc_f1_ncs_upper[3]]

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[3], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cvs[3], width, yerr = yerr_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del AB')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[4], lista_err_acc_f1_cvs_upper[4]]
yerr_ncs = [lista_err_acc_f1_ncs_lower[4], lista_err_acc_f1_ncs_upper[4]]

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[4], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cvs[4], width, yerr = yerr_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del MLP')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

####
# CV-NCS VS CV-CS

yerr_cvs = [lista_err_acc_f1_cvs_lower[0], lista_err_acc_f1_cvs_upper[0]]
yerr_cv_cs = [lista_err_acc_f1_cv_cs_lower[0], lista_err_acc_f1_cv_cs_upper[0]]

metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica)) # ottengo un array ([0,1])
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_cvs[0], width, yerr = yerr_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cv_cs[0], width, yerr = yerr_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del KNN')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


yerr_cvs = [lista_err_acc_f1_cvs_lower[1], lista_err_acc_f1_cvs_upper[1]]
yerr_cv_cs = [lista_err_acc_f1_cv_cs_lower[1], lista_err_acc_f1_cv_cs_upper[1]]


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_cvs[1], width, yerr = yerr_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cv_cs[1], width, yerr = yerr_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del SVM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[2], lista_err_acc_f1_cvs_upper[2]]
yerr_cv_cs = [lista_err_acc_f1_cv_cs_lower[2], lista_err_acc_f1_cv_cs_upper[2]]


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_cvs[2], width, yerr = yerr_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cv_cs[2], width, yerr = yerr_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del RF')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[3], lista_err_acc_f1_cvs_upper[3]]
yerr_cv_cs = [lista_err_acc_f1_cv_cs_lower[3], lista_err_acc_f1_cv_cs_upper[3]]


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_cvs[3], width, yerr = yerr_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cv_cs[3], width, yerr = yerr_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del AB')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_cvs = [lista_err_acc_f1_cvs_lower[4], lista_err_acc_f1_cvs_upper[4]]
yerr_cv_cs = [lista_err_acc_f1_cv_cs_lower[4], lista_err_acc_f1_cv_cs_upper[4]]


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_cvs[4], width, yerr = yerr_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cv_cs[4], width, yerr = yerr_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del MLP')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


##################
# NCS-CS

yerr_ncs = [lista_err_acc_f1_ncs_lower[0], lista_err_acc_f1_ncs_upper[0]]
yerr_cs = [lista_err_acc_f1_cs_lower[0], lista_err_acc_f1_cs_upper[0]]


metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica)) # ottengo un array ([0,1])
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[0], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cs[0], width, yerr = yerr_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del KNN')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


yerr_ncs = [lista_err_acc_f1_ncs_lower[1], lista_err_acc_f1_ncs_upper[1]]
yerr_cs = [lista_err_acc_f1_cs_lower[1], lista_err_acc_f1_cs_upper[1]]

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[1], width, yerr = yerr_ncs[1],capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cs[1], width, yerr = yerr_cs[1],capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del SVM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_ncs = [lista_err_acc_f1_ncs_lower[2], lista_err_acc_f1_ncs_upper[2]]
yerr_cs = [lista_err_acc_f1_cs_lower[2], lista_err_acc_f1_cs_upper[2]]


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[2], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cs[2], width, yerr = yerr_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del RF')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_ncs = [lista_err_acc_f1_ncs_lower[3], lista_err_acc_f1_ncs_upper[3]]
yerr_cs = [lista_err_acc_f1_cs_lower[3], lista_err_acc_f1_cs_upper[3]]

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[3], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cs[3], width, yerr = yerr_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del AB')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

yerr_ncs = [lista_err_acc_f1_ncs_lower[4], lista_err_acc_f1_ncs_upper[4]]
yerr_cs = [lista_err_acc_f1_cs_lower[4], lista_err_acc_f1_cs_upper[4]]

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, lista_mean_acc_f1_ncs[4], width, yerr = yerr_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, lista_mean_acc_f1_cs[4], width, yerr = yerr_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del MLP')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()



