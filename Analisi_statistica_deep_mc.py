import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# RETE CONV1D

with open('accuratezza_conv1d_cv_mc.pkl', 'rb') as file:
    accuratezza_modello_cvs = pickle.load(file)

with open('accuratezza_conv1d_ncs_mc.pkl' ,'rb') as file:
    accuratezza_modello_ncs = pickle.load(file)

with open('accuratezza_conv1d_cv_cs_mc.pkl' ,'rb') as file:
    accuratezza_modello_cv_cs = pickle.load(file)

with open('accuratezza_conv1d_cs_mc.pkl' ,'rb') as file:
    accuratezza_modello_cs = pickle.load(file)

acc_mean_cvs = np.mean(accuratezza_modello_cvs)
acc_mean_ncs = np.mean(accuratezza_modello_ncs)
acc_mean_cv_cs = np.mean(accuratezza_modello_cv_cs)
acc_mean_cs = np.mean(accuratezza_modello_cs)
print(f"valore medio accuratezza cvs",acc_mean_cvs)
print(f"valore medio accuratezza ncs",acc_mean_ncs)
print(f"valore medio accuratezza cv-cs",acc_mean_cv_cs)
print(f"valore medio accuratezza cs",acc_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(accuratezza_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT ACCURATEZZA')
axes[0].set_ylabel('CONV1D NCS')

stats.probplot(accuratezza_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV1D CVS')

stats.probplot(accuratezza_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV1D CV-CS')

stats.probplot(accuratezza_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV1D CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA ACCURATEZZA')
axes[0].set_ylabel('CONV1D NCS')

axes[1].hist(accuratezza_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV1D NCS')

axes[2].hist(accuratezza_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV1D CV-CS')

axes[3].hist(accuratezza_modello_cs, bins = 10)
axes[3].set_ylabel('CONV1D CS')

plt.show()

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modello_ncs)
axes[0].set_title('BOXPLOT ACCURATEZZA')
axes[0].set_ylabel('CONV1D NCS')

axes[1].boxplot(accuratezza_modello_cvs)
axes[1].set_ylabel('CONV1D CV')

axes[2].boxplot(accuratezza_modello_cv_cs)
axes[2].set_ylabel('CONV1D CV_CS')

axes[3].boxplot(accuratezza_modello_cs)
axes[3].set_ylabel('CONV1D CS')

plt.show()

#############
# boxplot

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "Accuratezza"
desired_stats_names = "Accuratezza (%)"

d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

d1 = {"Accuratezza":accuratezza_modello_cvs}
d2 = {"Accuratezza":accuratezza_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(accuratezza_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1D NCS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CVS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CV-CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1D CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1D CS",p_value)


# test shapiro

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1D NCS",p_value)

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CVS",p_value)


[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CV-CS",p_value)



[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CS",p_value)



alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:


    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_cvs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")




alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:


    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")






# suppongo la normalità

[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cvs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0",p_value)
    else:
        print(f"Test T, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_cvs,accuratezza_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_cvs,accuratezza_modello_cv_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cv_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


def intervalli_confidenza_conv1_acc(accuratezza):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    acc_ordin = np.sort(accuratezza)
    pmin = acc_ordin[round(index_low)]
    pmax = acc_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv1_acc(accuratezza_modello_ncs)
err_lower_ncs_acc = np.mean(accuratezza_modello_ncs) - p_min
err_upper_ncs_acc = p_max - np.mean(accuratezza_modello_ncs)
interval_conf_acc_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1_acc(accuratezza_modello_cvs)
err_lower_cvs_acc = np.mean(accuratezza_modello_cvs) - p_min
err_upper_cvs_acc = p_max - np.mean(accuratezza_modello_cvs)
interval_conf_acc_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1_acc(accuratezza_modello_cv_cs)
err_lower_cv_cs_acc = np.mean(accuratezza_modello_cv_cs) - p_min
err_upper_cv_cs_acc = p_max - np.mean(accuratezza_modello_cv_cs)
interval_conf_acc_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1_acc(accuratezza_modello_cs)
err_lower_cs_acc = np.mean(accuratezza_modello_cs) - p_min
err_upper_cs_acc = p_max - np.mean(accuratezza_modello_cs)
interval_conf_acc_cs = [p_min,p_max]

print(interval_conf_acc_cv_cs)
print(interval_conf_acc_cvs)
print(interval_conf_acc_ncs)
print(interval_conf_acc_cs)


with open('f1_conv1d_cv_mc.pkl', 'rb') as file:
    f1_modello_cvs = pickle.load(file)

with open('f1_conv1d_ncs_mc.pkl' ,'rb') as file:
    f1_modello_ncs = pickle.load(file)

with open('f1_conv1d_cv_cs_mc.pkl' ,'rb') as file:
    f1_modello_cv_cs = pickle.load(file)

with open('f1_conv1d_cs_mc.pkl' ,'rb') as file:
    f1_modello_cs = pickle.load(file)

f1_mean_ncs = np.mean(f1_modello_ncs)
f1_mean_cvs = np.mean(f1_modello_cvs)
f1_mean_cv_cs = np.mean(f1_modello_cv_cs)
f1_mean_cs = np.mean(f1_modello_cs)

print(f"valore medio f1 cvs",f1_mean_cvs)
print(f"valore medio f1 ncs",f1_mean_ncs)
print(f"valore medio f1 cv-cs",f1_mean_cv_cs)
print(f"valore medio f1 cs",f1_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(f1_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT F1')
axes[0].set_ylabel('CONV1D NCS')

stats.probplot(f1_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV1D CV')

stats.probplot(f1_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV1D CV-CS')

stats.probplot(f1_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV1D CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(f1_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA F1')
axes[0].set_ylabel('CONV1D NCS')

axes[1].hist(f1_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV1D CV')

axes[2].hist(f1_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV1D CV-CS')

axes[3].hist(f1_modello_cs, bins = 10)
axes[3].set_ylabel('CONV1D CS')

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(f1_modello_ncs)
axes[0].set_title('BOXPLOT F1')
axes[0].set_ylabel('CONV1D NCS')

axes[1].boxplot(f1_modello_cvs)
axes[1].set_ylabel('CONV1D CV')

axes[2].boxplot(f1_modello_cv_cs)
axes[2].set_ylabel('CONV1D CV_CS')

axes[3].boxplot(f1_modello_cs)
axes[3].set_ylabel('CONV1D CS')

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "F1_Score"
desired_stats_names = "F1_Score (%)"

d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_cvs}
d2 = {"F1_Score":f1_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(f1_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1D NCS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CVS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CS",p_value)

# test shapiro

[dist_norm, p_value] = stats.shapiro(f1_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1D NCS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CVS",p_value)


[dist_norm, p_value] = stats.shapiro(f1_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CV-CS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1D CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1D CS",p_value)




alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_cvs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')



alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


# suppongo la normalità

[stats_var, p_value] = stats.levene(f1_modello_ncs, f1_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cvs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)



def intervalli_confidenza_conv1_f1(f1):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    f1_ordin = np.sort(f1)
    pmin = f1_ordin[round(index_low)]
    pmax = f1_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv1_f1(f1_modello_ncs)
err_lower_ncs_f1 = np.mean(f1_modello_ncs) - p_min
err_upper_ncs_f1 = p_max - np.mean(f1_modello_ncs)
interval_conf_f1_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1_f1(f1_modello_cvs)
err_lower_cvs_f1 = np.mean(f1_modello_cvs) - p_min
err_upper_cvs_f1 = p_max - np.mean(f1_modello_cvs)
interval_conf_f1_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1_f1(f1_modello_cv_cs)
err_lower_cv_cs_f1 = np.mean(f1_modello_cv_cs) - p_min
err_upper_cv_cs_f1 = p_max - np.mean(f1_modello_cv_cs)
interval_conf_f1_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1_f1(f1_modello_cs)
err_lower_cs_f1 = np.mean(f1_modello_cs) - p_min
err_upper_cs_f1 = p_max - np.mean(f1_modello_cs)
interval_conf_f1_cs = [p_min,p_max]

print(interval_conf_f1_cv_cs)
print(interval_conf_f1_cvs)
print(interval_conf_f1_ncs)
print(interval_conf_f1_cs)

err_acc_f1_cvs_low = [err_lower_cvs_acc, err_lower_cvs_f1]
err_acc_f1_cvs_upper = [err_upper_cvs_acc, err_upper_cvs_f1]
err_cvs = [err_acc_f1_cvs_low, err_acc_f1_cvs_upper]

err_acc_f1_ncs_low = [err_lower_ncs_acc, err_lower_ncs_f1]
err_acc_f1_ncs_upper = [err_upper_ncs_acc, err_upper_ncs_f1]
err_ncs = [err_acc_f1_ncs_low, err_acc_f1_ncs_upper]

err_acc_f1_cv_cs_low = [err_lower_cv_cs_acc, err_lower_cv_cs_f1]
err_acc_f1_cv_cs_upper = [err_upper_cv_cs_acc, err_upper_cv_cs_f1]
err_cv_cs = [err_acc_f1_cv_cs_low, err_acc_f1_cv_cs_upper]

err_acc_f1_cs_low = [err_lower_cs_acc, err_lower_cs_f1]
err_acc_f1_cs_upper = [err_upper_cs_acc, err_upper_cs_f1]
err_cs = [err_acc_f1_cs_low, err_acc_f1_cs_upper]

value1 = np.mean(accuratezza_modello_cvs)
value2 = np.mean(f1_modello_cvs)
mean_acc_f1_cvs = [value1, value2]

value1 = np.mean(accuratezza_modello_ncs)
value2 = np.mean(f1_modello_ncs)
mean_acc_f1_ncs = [value1, value2]

value1 = np.mean(accuratezza_modello_cv_cs)
value2 = np.mean(f1_modello_cv_cs)
mean_acc_f1_cv_cs = [value1, value2]

value1 = np.mean(accuratezza_modello_cs)
value2 = np.mean(f1_modello_cs)
mean_acc_f1_cs = [value1, value2]


metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica))
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV1D')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cv_cs, width, yerr = err_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV1D')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cs, width, yerr = err_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV1D')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

####################

# rete conv2d


with open('accuratezza_conv2d_cv_mc.pkl', 'rb') as file:
    accuratezza_modello_cvs = pickle.load(file)

with open('accuratezza_conv2d_ncs_mc.pkl' ,'rb') as file:
    accuratezza_modello_ncs = pickle.load(file)

with open('accuratezza_conv2d_cs_cv_mc.pkl' ,'rb') as file:
    accuratezza_modello_cv_cs = pickle.load(file)

with open('accuratezza_conv2d_cs_mc.pkl' ,'rb') as file:
    accuratezza_modello_cs = pickle.load(file)

acc_mean_cvs = np.mean(accuratezza_modello_cvs)
acc_mean_ncs = np.mean(accuratezza_modello_ncs)
acc_mean_cv_cs = np.mean(accuratezza_modello_cv_cs)
acc_mean_cs = np.mean(accuratezza_modello_cs)
print(f"valore medio accuratezza cvs",acc_mean_cvs)
print(f"valore medio accuratezza ncs",acc_mean_ncs)
print(f"valore medio accuratezza cv-cs",acc_mean_cv_cs)
print(f"valore medio accuratezza cs",acc_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(accuratezza_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT ACCURATEZZA')
axes[0].set_ylabel('CONV2D NCS')

stats.probplot(accuratezza_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV2D CVS')

stats.probplot(accuratezza_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV2D CV-CS')

stats.probplot(accuratezza_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV2D CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA ACCURATEZZA')
axes[0].set_ylabel('CONV2D NCS')

axes[1].hist(accuratezza_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV2D NCS')

axes[2].hist(accuratezza_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV2D CV-CS')

axes[3].hist(accuratezza_modello_cs, bins = 10)
axes[3].set_ylabel('CONV2D CS')

plt.show()

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modello_ncs)
axes[0].set_title('BOXPLOT ACCURATEZZA')
axes[0].set_ylabel('CONV2D NCS')

axes[1].boxplot(accuratezza_modello_cvs)
axes[1].set_ylabel('CONV2D CV')

axes[2].boxplot(accuratezza_modello_cv_cs)
axes[2].set_ylabel('CONV2D CV_CS')

axes[3].boxplot(accuratezza_modello_cs)
axes[3].set_ylabel('CONV2D CS')

plt.show()

#############
# boxplot

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "Accuratezza"
desired_stats_names = "Accuratezza (%)"

d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

d1 = {"Accuratezza":accuratezza_modello_cvs}
d2 = {"Accuratezza":accuratezza_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(accuratezza_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2D NCS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CVS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CV-CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2D CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2D CS",p_value)


# test shapiro

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2D NCS",p_value)

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CVS",p_value)


[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CV-CS",p_value)



[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CS",p_value)




alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_cvs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")





alternative = ['greater','less','two-sided']
for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")




# suppongo la normalità

[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cvs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0",p_value)
    else:
        print(f"Test T, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_cvs,accuratezza_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_cvs,accuratezza_modello_cv_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



def intervalli_confidenza_conv2_acc(accuratezza):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    acc_ordin = np.sort(accuratezza)
    pmin = acc_ordin[round(index_low)]
    pmax = acc_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv2_acc(accuratezza_modello_ncs)
err_lower_ncs_acc = np.mean(accuratezza_modello_ncs) - p_min
err_upper_ncs_acc = p_max - np.mean(accuratezza_modello_ncs)
interval_conf_acc_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2_acc(accuratezza_modello_cvs)
err_lower_cvs_acc = np.mean(accuratezza_modello_cvs) - p_min
err_upper_cvs_acc = p_max - np.mean(accuratezza_modello_cvs)
interval_conf_acc_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2_acc(accuratezza_modello_cv_cs)
err_lower_cv_cs_acc = np.mean(accuratezza_modello_cv_cs) - p_min
err_upper_cv_cs_acc = p_max - np.mean(accuratezza_modello_cv_cs)
interval_conf_acc_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2_acc(accuratezza_modello_cs)
err_lower_cs_acc = np.mean(accuratezza_modello_cs) - p_min
err_upper_cs_acc = p_max - np.mean(accuratezza_modello_cs)
interval_conf_acc_cs = [p_min,p_max]

print(interval_conf_acc_cv_cs)
print(interval_conf_acc_cvs)
print(interval_conf_acc_ncs)
print(interval_conf_acc_cs)


with open('f1_conv2d_cv_mc.pkl', 'rb') as file:
    f1_modello_cvs = pickle.load(file)

with open('f1_conv2d_ncs_mc.pkl' ,'rb') as file:
    f1_modello_ncs = pickle.load(file)

with open('f1_conv2d_cs_cv_mc.pkl' ,'rb') as file:
    f1_modello_cv_cs = pickle.load(file)

with open('f1_conv2d_cs_mc.pkl' ,'rb') as file:
    f1_modello_cs = pickle.load(file)

f1_mean_ncs = np.mean(f1_modello_ncs)
f1_mean_cvs = np.mean(f1_modello_cvs)
f1_mean_cv_cs = np.mean(f1_modello_cv_cs)
f1_mean_cs = np.mean(f1_modello_cs)

print(f"valore medio f1 cvs",f1_mean_cvs)
print(f"valore medio f1 ncs",f1_mean_ncs)
print(f"valore medio f1 cv-cs",f1_mean_cv_cs)
print(f"valore medio f1 cs",f1_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(f1_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT F1')
axes[0].set_ylabel('CONV2D NCS')

stats.probplot(f1_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV2D CV')

stats.probplot(f1_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV2D CV-CS')

stats.probplot(f1_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV2D CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(f1_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA F1')
axes[0].set_ylabel('CONV2D NCS')

axes[1].hist(f1_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV2D CV')

axes[2].hist(f1_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV2D CV-CS')

axes[3].hist(f1_modello_cs, bins = 10)
axes[3].set_ylabel('CONV2D CS')

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(f1_modello_ncs)
axes[0].set_title('BOXPLOT F1')
axes[0].set_ylabel('CONV2D NCS')

axes[1].boxplot(f1_modello_cvs)
axes[1].set_ylabel('CONV2D CV')

axes[2].boxplot(f1_modello_cv_cs)
axes[2].set_ylabel('CONV2D CV_CS')

axes[3].boxplot(f1_modello_cs)
axes[3].set_ylabel('CONV2D CS')

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "F1_Score"
desired_stats_names = "F1_Score (%)"

d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_cvs}
d2 = {"F1_Score":f1_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(f1_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2D NCS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CVS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CS",p_value)

# test shapiro

[dist_norm, p_value] = stats.shapiro(f1_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2D NCS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CVS",p_value)


[dist_norm, p_value] = stats.shapiro(f1_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CV-CS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2D CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2D CS",p_value)



alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_cvs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')





for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')









# suppongo la normalità

[stats_var, p_value] = stats.levene(f1_modello_ncs, f1_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cvs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)


def intervalli_confidenza_conv2_f1(f1):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    f1_ordin = np.sort(f1)
    pmin = f1_ordin[round(index_low)]
    pmax = f1_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv2_f1(f1_modello_ncs)
err_lower_ncs_f1 = np.mean(f1_modello_ncs) - p_min
err_upper_ncs_f1 = p_max - np.mean(f1_modello_ncs)
interval_conf_f1_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2_f1(f1_modello_cvs)
err_lower_cvs_f1 = np.mean(f1_modello_cvs) - p_min
err_upper_cvs_f1 = p_max - np.mean(f1_modello_cvs)
interval_conf_f1_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2_f1(f1_modello_cv_cs)
err_lower_cv_cs_f1 = np.mean(f1_modello_cv_cs) - p_min
err_upper_cv_cs_f1 = p_max - np.mean(f1_modello_cv_cs)
interval_conf_f1_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2_f1(f1_modello_cs)
err_lower_cs_f1 = np.mean(f1_modello_cs) - p_min
err_upper_cs_f1 = p_max - np.mean(f1_modello_cs)
interval_conf_f1_cs = [p_min,p_max]

print(interval_conf_f1_cv_cs)
print(interval_conf_f1_cvs)
print(interval_conf_f1_ncs)
print(interval_conf_f1_cs)

err_acc_f1_cvs_low = [err_lower_cvs_acc, err_lower_cvs_f1]
err_acc_f1_cvs_upper = [err_upper_cvs_acc, err_upper_cvs_f1]
err_cvs = [err_acc_f1_cvs_low, err_acc_f1_cvs_upper]

err_acc_f1_ncs_low = [err_lower_ncs_acc, err_lower_ncs_f1]
err_acc_f1_ncs_upper = [err_upper_ncs_acc, err_upper_ncs_f1]
err_ncs = [err_acc_f1_ncs_low, err_acc_f1_ncs_upper]

err_acc_f1_cv_cs_low = [err_lower_cv_cs_acc, err_lower_cv_cs_f1]
err_acc_f1_cv_cs_upper = [err_upper_cv_cs_acc, err_upper_cv_cs_f1]
err_cv_cs = [err_acc_f1_cv_cs_low, err_acc_f1_cv_cs_upper]

err_acc_f1_cs_low = [err_lower_cs_acc, err_lower_cs_f1]
err_acc_f1_cs_upper = [err_upper_cs_acc, err_upper_cs_f1]
err_cs = [err_acc_f1_cs_low, err_acc_f1_cs_upper]

value1 = np.mean(accuratezza_modello_cvs)
value2 = np.mean(f1_modello_cvs)
mean_acc_f1_cvs = [value1, value2]

value1 = np.mean(accuratezza_modello_ncs)
value2 = np.mean(f1_modello_ncs)
mean_acc_f1_ncs = [value1, value2]

value1 = np.mean(accuratezza_modello_cv_cs)
value2 = np.mean(f1_modello_cv_cs)
mean_acc_f1_cv_cs = [value1, value2]

value1 = np.mean(accuratezza_modello_cs)
value2 = np.mean(f1_modello_cs)
mean_acc_f1_cs = [value1, value2]


metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica))
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV2D')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cv_cs, width, yerr = err_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV2D')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cs, width, yerr = err_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV2D')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

#############
# rete conv1dlstm


with open('accuratezza_conv1dlstm_cv_mc.pkl', 'rb') as file:
    accuratezza_modello_cvs = pickle.load(file)

with open('accuratezza_conv1dlstm_ncs_mc.pkl' ,'rb') as file:
    accuratezza_modello_ncs = pickle.load(file)

with open('accuratezza_conv1dlstm_cs_cv_mc.pkl' ,'rb') as file:
    accuratezza_modello_cv_cs = pickle.load(file)

with open('accuratezza_conv1dlstm_cs_mc.pkl' ,'rb') as file:
    accuratezza_modello_cs = pickle.load(file)

acc_mean_cvs = np.mean(accuratezza_modello_cvs)
acc_mean_ncs = np.mean(accuratezza_modello_ncs)
acc_mean_cv_cs = np.mean(accuratezza_modello_cv_cs)
acc_mean_cs = np.mean(accuratezza_modello_cs)
print(f"valore medio accuratezza cvs",acc_mean_cvs)
print(f"valore medio accuratezza ncs",acc_mean_ncs)
print(f"valore medio accuratezza cv-cs",acc_mean_cv_cs)
print(f"valore medio accuratezza cs",acc_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(accuratezza_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT ACCURATEZZA')
axes[0].set_ylabel('CONV1DLSTM NCS')

stats.probplot(accuratezza_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV1DLSTM CVS')

stats.probplot(accuratezza_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV1DLSTM CV-CS')

stats.probplot(accuratezza_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV1DLSTM CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA ACCURATEZZA')
axes[0].set_ylabel('CONV1DLSTM NCS')

axes[1].hist(accuratezza_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV1DLSTM NCS')

axes[2].hist(accuratezza_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV1DLSTM CV-CS')

axes[3].hist(accuratezza_modello_cs, bins = 10)
axes[3].set_ylabel('CONV1DLSTM CS')

plt.show()

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modello_ncs)
axes[0].set_title('BOXPLOT ACCURATEZZA')
axes[0].set_ylabel('CONV1DLSTM NCS')

axes[1].boxplot(accuratezza_modello_cvs)
axes[1].set_ylabel('CONV1DLSTM CV')

axes[2].boxplot(accuratezza_modello_cv_cs)
axes[2].set_ylabel('CONV1DLSTM CV_CS')

axes[3].boxplot(accuratezza_modello_cs)
axes[3].set_ylabel('CONV1DLSTM CS')

plt.show()

#############
# boxplot

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "Accuratezza"
desired_stats_names = "Accuratezza (%)"

d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

d1 = {"Accuratezza":accuratezza_modello_cvs}
d2 = {"Accuratezza":accuratezza_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(accuratezza_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1DLSTM NCS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CVS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CV-CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1DLSTM CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1DLSTM CS",p_value)


# test shapiro

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1DLSTM NCS",p_value)

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CVS",p_value)


[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CV-CS",p_value)



[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CS",p_value)



alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_cvs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")




for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")





# suppongo la normalità

[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cvs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0",p_value)
    else:
        print(f"Test T, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_cvs,accuratezza_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_cvs,accuratezza_modello_cv_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



def intervalli_confidenza_conv1lstm_acc(accuratezza):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    acc_ordin = np.sort(accuratezza)
    pmin = acc_ordin[round(index_low)]
    pmax = acc_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv1lstm_acc(accuratezza_modello_ncs)
err_lower_ncs_acc = np.mean(accuratezza_modello_ncs) - p_min
err_upper_ncs_acc = p_max - np.mean(accuratezza_modello_ncs)
interval_conf_acc_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1lstm_acc(accuratezza_modello_cvs)
err_lower_cvs_acc = np.mean(accuratezza_modello_cvs) - p_min
err_upper_cvs_acc = p_max - np.mean(accuratezza_modello_cvs)
interval_conf_acc_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1lstm_acc(accuratezza_modello_cv_cs)
err_lower_cv_cs_acc = np.mean(accuratezza_modello_cv_cs) - p_min
err_upper_cv_cs_acc = p_max - np.mean(accuratezza_modello_cv_cs)
interval_conf_acc_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1lstm_acc(accuratezza_modello_cs)
err_lower_cs_acc = np.mean(accuratezza_modello_cs) - p_min
err_upper_cs_acc = p_max - np.mean(accuratezza_modello_cs)
interval_conf_acc_cs = [p_min,p_max]

print(interval_conf_acc_cv_cs)
print(interval_conf_acc_cvs)
print(interval_conf_acc_ncs)
print(interval_conf_acc_cs)


with open('f1_conv1dlstm_cv_mc.pkl', 'rb') as file:
    f1_modello_cvs = pickle.load(file)

with open('f1_conv1dlstm_ncs_mc.pkl' ,'rb') as file:
    f1_modello_ncs = pickle.load(file)

with open('f1_conv1dlstm_cs_cv_mc.pkl' ,'rb') as file:
    f1_modello_cv_cs = pickle.load(file)

with open('f1_conv1dlstm_cs_mc.pkl' ,'rb') as file:
    f1_modello_cs = pickle.load(file)

f1_mean_ncs = np.mean(f1_modello_ncs)
f1_mean_cvs = np.mean(f1_modello_cvs)
f1_mean_cv_cs = np.mean(f1_modello_cv_cs)
f1_mean_cs = np.mean(f1_modello_cs)

print(f"valore medio f1 cvs",f1_mean_cvs)
print(f"valore medio f1 ncs",f1_mean_ncs)
print(f"valore medio f1 cv-cs",f1_mean_cv_cs)
print(f"valore medio f1 cs",f1_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(f1_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT F1')
axes[0].set_ylabel('CONV1DLSTM NCS')

stats.probplot(f1_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV1DLSTM CV')

stats.probplot(f1_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV1DLSTM CV-CS')

stats.probplot(f1_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV1DLSTM CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(f1_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA F1')
axes[0].set_ylabel('CONV1DLSTM NCS')

axes[1].hist(f1_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV1DLSTM CV')

axes[2].hist(f1_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV1DLSTM CV-CS')

axes[3].hist(f1_modello_cs, bins = 10)
axes[3].set_ylabel('CONV1DLSTM CS')

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(f1_modello_ncs)
axes[0].set_title('BOXPLOT F1')
axes[0].set_ylabel('CONV1DLSTM NCS')

axes[1].boxplot(f1_modello_cvs)
axes[1].set_ylabel('CONV1DLSTM CV')

axes[2].boxplot(f1_modello_cv_cs)
axes[2].set_ylabel('CONV1DLSTM CV_CS')

axes[3].boxplot(f1_modello_cs)
axes[3].set_ylabel('CONV1DLSTM CS')

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "F1_Score"
desired_stats_names = "F1_Score (%)"

d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_cvs}
d2 = {"F1_Score":f1_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(f1_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1DLSTM NCS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CVS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CS",p_value)

# test shapiro

[dist_norm, p_value] = stats.shapiro(f1_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV1DLSTM NCS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CVS",p_value)


[dist_norm, p_value] = stats.shapiro(f1_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CV-CS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV1DLSTM CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV1DLSTM CS",p_value)




alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_cvs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')





for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')








# suppongo la normalità

[stats_var, p_value] = stats.levene(f1_modello_ncs, f1_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cvs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)


def intervalli_confidenza_conv1lstm_f1(f1):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    f1_ordin = np.sort(f1)
    pmin = f1_ordin[round(index_low)]
    pmax = f1_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv1lstm_f1(f1_modello_ncs)
err_lower_ncs_f1 = np.mean(f1_modello_ncs) - p_min
err_upper_ncs_f1 = p_max - np.mean(f1_modello_ncs)
interval_conf_f1_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1lstm_f1(f1_modello_cvs)
err_lower_cvs_f1 = np.mean(f1_modello_cvs) - p_min
err_upper_cvs_f1 = p_max - np.mean(f1_modello_cvs)
interval_conf_f1_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1lstm_f1(f1_modello_cv_cs)
err_lower_cv_cs_f1 = np.mean(f1_modello_cv_cs) - p_min
err_upper_cv_cs_f1 = p_max - np.mean(f1_modello_cv_cs)
interval_conf_f1_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv1lstm_f1(f1_modello_cs)
err_lower_cs_f1 = np.mean(f1_modello_cs) - p_min
err_upper_cs_f1 = p_max - np.mean(f1_modello_cs)
interval_conf_f1_cs = [p_min,p_max]

print(interval_conf_f1_cv_cs)
print(interval_conf_f1_cvs)
print(interval_conf_f1_ncs)
print(interval_conf_f1_cs)

err_acc_f1_cvs_low = [err_lower_cvs_acc, err_lower_cvs_f1]
err_acc_f1_cvs_upper = [err_upper_cvs_acc, err_upper_cvs_f1]
err_cvs = [err_acc_f1_cvs_low, err_acc_f1_cvs_upper]

err_acc_f1_ncs_low = [err_lower_ncs_acc, err_lower_ncs_f1]
err_acc_f1_ncs_upper = [err_upper_ncs_acc, err_upper_ncs_f1]
err_ncs = [err_acc_f1_ncs_low, err_acc_f1_ncs_upper]

err_acc_f1_cv_cs_low = [err_lower_cv_cs_acc, err_lower_cv_cs_f1]
err_acc_f1_cv_cs_upper = [err_upper_cv_cs_acc, err_upper_cv_cs_f1]
err_cv_cs = [err_acc_f1_cv_cs_low, err_acc_f1_cv_cs_upper]

err_acc_f1_cs_low = [err_lower_cs_acc, err_lower_cs_f1]
err_acc_f1_cs_upper = [err_upper_cs_acc, err_upper_cs_f1]
err_cs = [err_acc_f1_cs_low, err_acc_f1_cs_upper]

value1 = np.mean(accuratezza_modello_cvs)
value2 = np.mean(f1_modello_cvs)
mean_acc_f1_cvs = [value1, value2]

value1 = np.mean(accuratezza_modello_ncs)
value2 = np.mean(f1_modello_ncs)
mean_acc_f1_ncs = [value1, value2]

value1 = np.mean(accuratezza_modello_cv_cs)
value2 = np.mean(f1_modello_cv_cs)
mean_acc_f1_cv_cs = [value1, value2]

value1 = np.mean(accuratezza_modello_cs)
value2 = np.mean(f1_modello_cs)
mean_acc_f1_cs = [value1, value2]


metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica))
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV1DLSTM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cv_cs, width, yerr = err_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV1DLSTM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cs, width, yerr = err_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV1DLSTM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()

#############
# rete conv2dlstm


with open('accuratezza_conv2dlstm_cv_mc.pkl', 'rb') as file:
    accuratezza_modello_cvs = pickle.load(file)

with open('accuratezza_conv2dlstm_ncs_mc.pkl' ,'rb') as file:
    accuratezza_modello_ncs = pickle.load(file)

with open('accuratezza_conv2dlstm_cs_cv_mc.pkl' ,'rb') as file:
    accuratezza_modello_cv_cs_mc = pickle.load(file)

with open('accuratezza_conv2dlstm_cs_cv_mc1.pkl', 'rb') as file:
    accuratezza_modello_cv_cs_mc1 = pickle.load(file)

accuratezza_modello_cv_cs = np.concatenate((accuratezza_modello_cv_cs_mc, accuratezza_modello_cv_cs_mc1))

with open('accuratezza_conv2dlstm_cs_mc.pkl' ,'rb') as file:
    accuratezza_modello_cs_mc = pickle.load(file)

with open('accuratezza_conv2dlstm_cs_mc1.pkl' ,'rb') as file:
    accuratezza_modello_cs_mc1 = pickle.load(file)

accuratezza_modello_cs = np.concatenate((accuratezza_modello_cs_mc, accuratezza_modello_cs_mc1))


acc_mean_cvs = np.mean(accuratezza_modello_cvs)
acc_mean_ncs = np.mean(accuratezza_modello_ncs)
acc_mean_cv_cs = np.mean(accuratezza_modello_cv_cs)
acc_mean_cs = np.mean(accuratezza_modello_cs)
print(f"valore medio accuratezza cvs",acc_mean_cvs)
print(f"valore medio accuratezza ncs",acc_mean_ncs)
print(f"valore medio accuratezza cv-cs",acc_mean_cv_cs)
print(f"valore medio accuratezza cs",acc_mean_cs)



fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(accuratezza_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT ACCURATEZZA')
axes[0].set_ylabel('CONV2DLSTM NCS')

stats.probplot(accuratezza_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV2DLSTM CVS')

stats.probplot(accuratezza_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV2DLSTM CV-CS')

stats.probplot(accuratezza_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV2DLSTM CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(accuratezza_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA ACCURATEZZA')
axes[0].set_ylabel('CONV2DLSTM NCS')

axes[1].hist(accuratezza_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV2DLSTM NCS')

axes[2].hist(accuratezza_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV2DLSTM CV-CS')

axes[3].hist(accuratezza_modello_cs, bins = 10)
axes[3].set_ylabel('CONV2DLSTM CS')

plt.show()

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(accuratezza_modello_ncs)
axes[0].set_title('BOXPLOT ACCURATEZZA')
axes[0].set_ylabel('CONV2DLSTM NCS')

axes[1].boxplot(accuratezza_modello_cvs)
axes[1].set_ylabel('CONV2DLSTM CV')

axes[2].boxplot(accuratezza_modello_cv_cs)
axes[2].set_ylabel('CONV2DLSTM CV_CS')

axes[3].boxplot(accuratezza_modello_cs)
axes[3].set_ylabel('CONV2DLSTM CS')

plt.show()

#############
# boxplot

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "Accuratezza"
desired_stats_names = "Accuratezza (%)"

d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"Accuratezza":accuratezza_modello_ncs}
d2 = {"Accuratezza":accuratezza_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

d1 = {"Accuratezza":accuratezza_modello_cvs}
d2 = {"Accuratezza":accuratezza_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(accuratezza_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2DLSTM NCS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CVS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CV-CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2DLSTM CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(accuratezza_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2DLSTM CS",p_value)


# test shapiro

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2DLSTM NCS",p_value)

[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CVS",p_value)


[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CV-CS",p_value)


[dist_norm, p_value] = stats.shapiro(accuratezza_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CS",p_value)



alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_cvs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")


for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")





for alt in alternative:

    [stats_ugual, p_value] = stats.mannwhitneyu(accuratezza_modello_ncs,accuratezza_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")




# suppongo la normalità

[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cvs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0",p_value)
    else:
        print(f"Test T, accetto H0",p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_cvs,accuratezza_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_cvs,accuratezza_modello_cv_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")



[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")




[stats_var, p_value] = stats.levene(accuratezza_modello_ncs,accuratezza_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

for alt in alternative:
    [stats_uguag, p_value] = stats.ttest_ind(accuratezza_modello_ncs,accuratezza_modello_cv_cs, alternative=alt)

    if p_value < alpha:
        print(f"Test T, rifiuto H0", p_value)
    else:
        print(f"Test T, accetto H0", p_value)

    if p_value < alpha:
        print(f"condizione verificata per {alt}")







def intervalli_confidenza_conv2lstm_acc(accuratezza):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    acc_ordin = np.sort(accuratezza)
    pmin = acc_ordin[round(index_low)]
    pmax = acc_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv2lstm_acc(accuratezza_modello_ncs)
err_lower_ncs_acc = np.mean(accuratezza_modello_ncs) - p_min
err_upper_ncs_acc = p_max - np.mean(accuratezza_modello_ncs)
interval_conf_acc_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2lstm_acc(accuratezza_modello_cvs)
err_lower_cvs_acc = np.mean(accuratezza_modello_cvs) - p_min
err_upper_cvs_acc = p_max - np.mean(accuratezza_modello_cvs)
interval_conf_acc_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2lstm_acc(accuratezza_modello_cv_cs)
err_lower_cv_cs_acc = np.mean(accuratezza_modello_cv_cs) - p_min
err_upper_cv_cs_acc = p_max - np.mean(accuratezza_modello_cv_cs)
interval_conf_acc_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2lstm_acc(accuratezza_modello_cs)
err_lower_cs_acc = np.mean(accuratezza_modello_cs) - p_min
err_upper_cs_acc = p_max - np.mean(accuratezza_modello_cs)
interval_conf_acc_cs = [p_min,p_max]

print(interval_conf_acc_cv_cs)
print(interval_conf_acc_cvs)
print(interval_conf_acc_ncs)
print(interval_conf_acc_cs)


with open('f1_conv2dlstm_cv_mc.pkl', 'rb') as file:
    f1_modello_cvs = pickle.load(file)

with open('f1_conv2dlstm_ncs_mc.pkl' ,'rb') as file:
    f1_modello_ncs = pickle.load(file)

with open('f1_conv2dlstm_cs_cv_mc.pkl' ,'rb') as file:
    f1_modello_cv_cs_mc = pickle.load(file)

with open('f1_conv2dlstm_cs_cv_mc1.pkl' ,'rb') as file:
    f1_modello_cv_cs_mc1 = pickle.load(file)

f1_modello_cv_cs = np.concatenate((f1_modello_cv_cs_mc, f1_modello_cv_cs_mc1))

with open('f1_conv2dlstm_cs_mc.pkl' ,'rb') as file:
    f1_modello_cs_mc = pickle.load(file)

with open('f1_conv2dlstm_cs_mc1.pkl', 'rb') as file:
    f1_modello_cs_mc1 = pickle.load(file)

f1_modello_cs = np.concatenate((f1_modello_cs_mc, f1_modello_cs_mc1))

f1_mean_ncs = np.mean(f1_modello_ncs)
f1_mean_cvs = np.mean(f1_modello_cvs)
f1_mean_cv_cs = np.mean(f1_modello_cv_cs)
f1_mean_cs = np.mean(f1_modello_cs)

print(f"valore medio f1 cvs",f1_mean_cvs)
print(f"valore medio f1 ncs",f1_mean_ncs)
print(f"valore medio f1 cv-cs",f1_mean_cv_cs)
print(f"valore medio f1 cs",f1_mean_cs)

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

stats.probplot(f1_modello_ncs, dist='norm', plot=axes[0])
axes[0].set_title('QQ-PLOT F1')
axes[0].set_ylabel('CONV2DLSTM NCS')

stats.probplot(f1_modello_cvs, dist='norm', plot=axes[1])
axes[1].set_ylabel('CONV2DLSTM CV')

stats.probplot(f1_modello_cv_cs, dist='norm', plot=axes[2])
axes[2].set_ylabel('CONV2DLSTM CV-CS')

stats.probplot(f1_modello_cs, dist='norm', plot=axes[3])
axes[3].set_ylabel('CONV2DLSTM CS')

plt.show()

fig, axes = plt.subplots(nrows = 4,ncols = 1, sharex = True)

axes[0].hist(f1_modello_ncs, bins = 10)
axes[0].set_title('ISTOGRAMMA F1')
axes[0].set_ylabel('CONV2DLSTM NCS')

axes[1].hist(f1_modello_cvs, bins = 10)
axes[1].set_ylabel('CONV2DLSTM CV')

axes[2].hist(f1_modello_cv_cs, bins = 10)
axes[2].set_ylabel('CONV2DLSTM CV-CS')

axes[3].hist(f1_modello_cs, bins = 10)
axes[3].set_ylabel('CONV2DLSTM CS')

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)

axes[0].boxplot(f1_modello_ncs)
axes[0].set_title('BOXPLOT F1')
axes[0].set_ylabel('CONV2DLSTM NCS')

axes[1].boxplot(f1_modello_cvs)
axes[1].set_ylabel('CONV2DLSTM CV')

axes[2].boxplot(f1_modello_cv_cs)
axes[2].set_ylabel('CONV2DLSTM CV_CS')

axes[3].boxplot(f1_modello_cs)
axes[3].set_ylabel('CONV2DLSTM CS')

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
    plt.title("Statistica del test set")
    plt.tight_layout()
    plt.show()

desired_stats = "F1_Score"
desired_stats_names = "F1_Score (%)"

d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cvs}

descr1 = "Modello NCV"
descr2 = "Modello CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_ncs}
d2 = {"F1_Score":f1_modello_cs}

descr1 = "Modello NCV"
descr2 = "Modello NCV-CS"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)


d1 = {"F1_Score":f1_modello_cvs}
d2 = {"F1_Score":f1_modello_cv_cs}

descr1 = "Modello CV"
descr2 = "Modello CS_CV"

plot_statistics(d1,d2,desired_stats, desired_stats_names,descr1,descr2)

# test kolmogorov-smirnov

[dist_norm, p_value] = stats.kstest(f1_modello_ncs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2DLSTM NCS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cvs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CVS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cv_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CV-CS",p_value)

[dist_norm, p_value] = stats.kstest(f1_modello_cs, 'norm')

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CS",p_value)

# test shapiro

[dist_norm, p_value] = stats.shapiro(f1_modello_ncs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM NCS",p_value)
elif p_value > 0.05:
        print(f"accetto l'ipotesi per CONV2DLSTM NCS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cvs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CVS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CVS",p_value)


[dist_norm, p_value] = stats.shapiro(f1_modello_cv_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CV-CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CV-CS",p_value)

[dist_norm, p_value] = stats.shapiro(f1_modello_cs)

if p_value < 0.05:
    print(f"rifiuto l'ipotesi nulla per CONV2DLSTM CS",p_value)
elif p_value > 0.05:
    print(f"accetto l'ipotesi per CONV2DLSTM CS",p_value)


alpha = 0.05
alternative = ['greater','less','two-sided']
for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cvs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_cvs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')


for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')



for alt in alternative:
    [stats_uguag, p_value] = stats.mannwhitneyu(f1_modello_ncs,f1_modello_cv_cs,alternative=alt)

    if p_value < alpha:
        print(f"Test Mann-Whitney, rifiuto H0",p_value)
    else:
        print(f"Test Mann-Whitney, accetto H0",p_value)

    if p_value < alpha:
        print(f'condizione verificata per {alt}')



# suppongo la normalità

[stats_var, p_value] = stats.levene(f1_modello_ncs, f1_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cvs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)



[stats_var, p_value] = stats.levene(f1_modello_ncs, f1_modello_cvs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cvs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)



[stats_var, p_value] = stats.levene(f1_modello_cvs, f1_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cv_cs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)



[stats_var, p_value] = stats.levene(f1_modello_ncs, f1_modello_cv_cs)

alpha = 0.05
if p_value < alpha:
    print(f"Test Levene, rifiuto H0",p_value)
else:
    print(f"Test Levene, accetto H0",p_value)

[stats_uguag, p_value] = stats.ttest_ind(f1_modello_ncs,f1_modello_cs)

if p_value < alpha:
    print(f"Test T, rifiuto H0", p_value)
else:
    print(f"Test T, accetto H0", p_value)


def intervalli_confidenza_conv2lstm_f1(f1):
    alpha = 0.05
    index_low = (alpha / 2) * 100
    index_high = (1 - alpha / 2) * 100
    f1_ordin = np.sort(f1)
    pmin = f1_ordin[round(index_low)]
    pmax = f1_ordin[round(index_high)]

    return pmin, pmax

[p_min, p_max] = intervalli_confidenza_conv2lstm_f1(f1_modello_ncs)
err_lower_ncs_f1 = np.mean(f1_modello_ncs) - p_min
err_upper_ncs_f1 = p_max - np.mean(f1_modello_ncs)
interval_conf_f1_ncs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2lstm_f1(f1_modello_cvs)
err_lower_cvs_f1 = np.mean(f1_modello_cvs) - p_min
err_upper_cvs_f1 = p_max - np.mean(f1_modello_cvs)
interval_conf_f1_cvs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2lstm_f1(f1_modello_cv_cs)
err_lower_cv_cs_f1 = np.mean(f1_modello_cv_cs) - p_min
err_upper_cv_cs_f1 = p_max - np.mean(f1_modello_cv_cs)
interval_conf_f1_cv_cs = [p_min,p_max]

[p_min, p_max] = intervalli_confidenza_conv2lstm_f1(f1_modello_cs)
err_lower_cs_f1 = np.mean(f1_modello_cs) - p_min
err_upper_cs_f1 = p_max - np.mean(f1_modello_cs)
interval_conf_f1_cs = [p_min,p_max]

print(interval_conf_f1_cv_cs)
print(interval_conf_f1_cvs)
print(interval_conf_f1_ncs)
print(interval_conf_f1_cs)

err_acc_f1_cvs_low = [err_lower_cvs_acc, err_lower_cvs_f1]
err_acc_f1_cvs_upper = [err_upper_cvs_acc, err_upper_cvs_f1]
err_cvs = [err_acc_f1_cvs_low, err_acc_f1_cvs_upper]

err_acc_f1_ncs_low = [err_lower_ncs_acc, err_lower_ncs_f1]
err_acc_f1_ncs_upper = [err_upper_ncs_acc, err_upper_ncs_f1]
err_ncs = [err_acc_f1_ncs_low, err_acc_f1_ncs_upper]

err_acc_f1_cv_cs_low = [err_lower_cv_cs_acc, err_lower_cv_cs_f1]
err_acc_f1_cv_cs_upper = [err_upper_cv_cs_acc, err_upper_cv_cs_f1]
err_cv_cs = [err_acc_f1_cv_cs_low, err_acc_f1_cv_cs_upper]

err_acc_f1_cs_low = [err_lower_cs_acc, err_lower_cs_f1]
err_acc_f1_cs_upper = [err_upper_cs_acc, err_upper_cs_f1]
err_cs = [err_acc_f1_cs_low, err_acc_f1_cs_upper]

value1 = np.mean(accuratezza_modello_cvs)
value2 = np.mean(f1_modello_cvs)
mean_acc_f1_cvs = [value1, value2]

value1 = np.mean(accuratezza_modello_ncs)
value2 = np.mean(f1_modello_ncs)
mean_acc_f1_ncs = [value1, value2]

value1 = np.mean(accuratezza_modello_cv_cs)
value2 = np.mean(f1_modello_cv_cs)
mean_acc_f1_cv_cs = [value1, value2]

value1 = np.mean(accuratezza_modello_cs)
value2 = np.mean(f1_modello_cs)
mean_acc_f1_cs = [value1, value2]


metrica = ['Test Accuracy','Test F1-Score']
x = np.arange(len(metrica))
width = 0.35

fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV2DLSTM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_cvs, width, yerr = err_cvs,capsize = 5,label='modello CV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cv_cs, width, yerr = err_cv_cs,capsize = 5,label='modello CS-CV')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV2DLSTM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize=(8,6))
bars1 = ax.bar(x - width/2, mean_acc_f1_ncs, width, yerr = err_ncs,capsize = 5,label='modello NCV')
bars2 = ax.bar(x + width/2, mean_acc_f1_cs, width, yerr = err_cs,capsize = 5,label='modello NCV-CS')

ax.set_xlabel('Metric Type')
ax.set_ylabel('Value con 95% CI')
ax.set_title('Statistica del CONV2DLSTM')
ax.set_xticks(x)
ax.legend(loc = 'upper center')
ax.set_xticklabels(metrica)

plt.tight_layout()
plt.show()