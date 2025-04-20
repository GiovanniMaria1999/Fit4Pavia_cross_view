import pickle
import numpy as np


# analisi statistica ML

with open('confusion_matrix_ncs.pkl', 'rb') as f:
    confusion_matrix_ncs = pickle.load(f)

with open('confusion_matrix_cs.pkl', 'rb') as f:
    confusion_matrix_cs = pickle.load(f)

with open('confusion_matrix_cvs.pkl', 'rb') as f:
    confusion_matrix_cv = pickle.load(f)

with open('confusion_matrix_cv_cs.pkl', 'rb') as f:
    confusion_matrix_cv_cs = pickle.load(f)

# calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC

acc_ml_ncs = []
acc_ml_cs = []
acc_ml_cv = []
acc_ml_cs_cv = []

for i in range(5):

    acc_ncs = (confusion_matrix_ncs[i][0,0] + confusion_matrix_ncs[i][1,1])/(confusion_matrix_ncs[i][0,0] + confusion_matrix_ncs[i][1,0] + confusion_matrix_ncs[i][0,1] + confusion_matrix_ncs[i][1,1])
    acc_ml_ncs.append(acc_ncs)

    acc_cs = (confusion_matrix_cs[i][0, 0] + confusion_matrix_cs[i][1, 1]) / (confusion_matrix_cs[i][0, 0] + confusion_matrix_cs[i][1, 0] + confusion_matrix_cs[i][0, 1] +confusion_matrix_cs[i][1, 1])
    acc_ml_cs.append(acc_cs)

    acc_cv = (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[i][1, 1]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[i][1, 0] + confusion_matrix_cv[i][0, 1] +confusion_matrix_cv[i][1, 1])
    acc_ml_cv.append(acc_cv)

    acc_cs_cv = (confusion_matrix_cv_cs[i][0, 0] + confusion_matrix_cv_cs[i][1, 1]) / (confusion_matrix_cv_cs[i][0, 0] + confusion_matrix_cv_cs[i][1, 0] + confusion_matrix_cv_cs[i][0, 1] +confusion_matrix_cv_cs[i][1, 1])
    acc_ml_cs_cv.append(acc_cs_cv)


spec_ml_ncs = []
spec_ml_cs = []
spec_ml_cv = []
spec_ml_cs_cv = []

for i in range(5):

    spec_ncs = (confusion_matrix_ncs[i][1,1])/(confusion_matrix_ncs[i][0,1] + confusion_matrix_ncs[i][1,1])
    spec_ml_ncs.append(spec_ncs)

    spec_cs = (confusion_matrix_cs[i][1, 1]) / (confusion_matrix_cs[i][0, 1] +confusion_matrix_cs[i][1, 1])
    spec_ml_cs.append(spec_cs)

    spec_cv = (confusion_matrix_cv[i][1, 1]) / (confusion_matrix_cv[i][0, 1] +confusion_matrix_cv[i][1, 1])
    spec_ml_cv.append(spec_cv)

    spec_cs_cv = (confusion_matrix_cv_cs[i][1, 1]) / (confusion_matrix_cv_cs[i][0, 1] +confusion_matrix_cv_cs[i][1, 1])
    spec_ml_cs_cv.append(spec_cs_cv)



sens_ml_ncs = []
sens_ml_cs = []
sens_ml_cv = []
sens_ml_cs_cv = []

for i in range(5):

    sens_ncs = (confusion_matrix_ncs[i][0,0])/(confusion_matrix_ncs[i][0,0] + confusion_matrix_ncs[i][1,0])
    sens_ml_ncs.append(sens_ncs)

    sens_cs = (confusion_matrix_cs[i][0, 0]) / (confusion_matrix_cs[i][0, 0] + confusion_matrix_cs[i][1, 0])
    sens_ml_cs.append(sens_cs)

    sens_cv = (confusion_matrix_cv[i][0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[i][1, 0])
    sens_ml_cv.append(sens_cv)

    sens_cs_cv = (confusion_matrix_cv_cs[i][0, 0]) / (confusion_matrix_cv_cs[i][0, 0] + confusion_matrix_cv_cs[i][1, 0])
    sens_ml_cs_cv.append(sens_cs_cv)



recal_ml_ncs = []
recal_ml_cs = []
recal_ml_cv = []
recal_ml_cs_cv = []

for i in range(5):

    recal_ncs = (confusion_matrix_ncs[i][0,0])/(confusion_matrix_ncs[i][0,0] + confusion_matrix_ncs[i][1,0])
    recal_ml_ncs.append(recal_ncs)

    recal_cs = (confusion_matrix_cs[i][0,0]) / (confusion_matrix_cs[i][0,0] + confusion_matrix_cs[i][1, 0])
    recal_ml_cs.append(recal_cs)

    recal_cv = (confusion_matrix_cv[i][0,0]) / (confusion_matrix_cv[i][0,0] + confusion_matrix_cv[i][1, 0])
    recal_ml_cv.append(recal_cv)

    recal_cs_cv = (confusion_matrix_cv_cs[i][0,0]) / (confusion_matrix_cv_cs[i][0,0] + confusion_matrix_cv_cs[i][1, 0])
    recal_ml_cs_cv.append(recal_cs_cv)



prec_ml_ncs = []
prec_ml_cs = []
prec_ml_cv = []
prec_ml_cs_cv = []

for i in range(5):

    prec_ncs = (confusion_matrix_ncs[i][0,0])/(confusion_matrix_ncs[i][0,0] + confusion_matrix_ncs[i][0,1])
    prec_ml_ncs.append(prec_ncs)

    prec_cs = (confusion_matrix_cs[i][0,0]) / (confusion_matrix_cs[i][0,0] + confusion_matrix_cs[i][0,1])
    prec_ml_cs.append(prec_cs)

    prec_cv = (confusion_matrix_cv[i][0,0]) / (confusion_matrix_cv[i][0,0] + confusion_matrix_cv[i][0,1])
    prec_ml_cv.append(prec_cv)

    prec_cs_cv = (confusion_matrix_cv_cs[i][0,0]) / (confusion_matrix_cv_cs[i][0,0] + confusion_matrix_cv_cs[i][0,1])
    prec_ml_cs_cv.append(prec_cs_cv)

f_ml_ncs = []
f_ml_cs = []
f_ml_cv = []
f_ml_cs_cv = []

for i in range(5):

    f_ncs = 2 * recal_ml_ncs[i]*prec_ml_ncs[i]/(recal_ml_ncs[i] + prec_ml_ncs[i])
    f_ml_ncs.append(f_ncs)

    f_cs = 2 * recal_ml_cs[i] * prec_ml_cs[i] / (recal_ml_cs[i] + prec_ml_cs[i])
    f_ml_cs.append(f_cs)

    f_cv = 2 * recal_ml_cv[i] * prec_ml_cv[i] / (recal_ml_cv[i] + prec_ml_cv[i])
    f_ml_cv.append(f_cv)

    f_cs_cv = 2 * recal_ml_cs_cv[i] * prec_ml_cs_cv[i] / (recal_ml_cs_cv[i] + prec_ml_cs_cv[i])
    f_ml_cs_cv.append(f_cs_cv)


mcc_ml_ncs = []
mcc_ml_cs = []
mcc_ml_cv = []
mcc_ml_cs_cv = []

for i in range(5):

    mcc_ncs = (confusion_matrix_ncs[i][0,0]*confusion_matrix_ncs[i][1,1] - confusion_matrix_ncs[i][0,1]*confusion_matrix_ncs[i][1,0])/np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])*(confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1])*(confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])*(confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_ml_ncs.append(mcc_ncs)

    mcc_cs = (confusion_matrix_cs[i][0, 0] * confusion_matrix_cs[i][1, 1] - confusion_matrix_cs[i][0, 1]*confusion_matrix_cs[i][1, 0])/np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])*(confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1])*(confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])*(confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_ml_cs.append(mcc_cs)

    mcc_cv = (confusion_matrix_cv[i][0, 0] * confusion_matrix_cv[i][1, 1] - confusion_matrix_cv[i][0, 1] *confusion_matrix_cv[i][1, 0])/np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])*(confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1])*(confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])*(confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_ml_cv.append(mcc_cv)

    mcc_cs_cv = (confusion_matrix_cv_cs[i][0, 0] * confusion_matrix_cv_cs[i][1, 1] - confusion_matrix_cv_cs[i][0, 1] *confusion_matrix_cv_cs[i][1, 0])/np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])*(confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1])*(confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])*(confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_ml_cs_cv.append(mcc_cs_cv)




# analisi statistica DL

# conv1d
    with open('conf_matrix_conv1d_ncs.pkl', 'rb') as f:
        confusion_matrix_ncs = pickle.load(f)

    with open('conf_matrix_conv1d_cs.pkl', 'rb') as f:
        confusion_matrix_cs = pickle.load(f)

    with open('conf_matrix_conv1d_cvs.pkl', 'rb') as f:
        confusion_matrix_cv = pickle.load(f)

    with open('conf_matrix_conv1d_cv_cs.pkl', 'rb') as f:
        confusion_matrix_cv_cs = pickle.load(f)

    # calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC


    acc_ncs = (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[0, 1] +confusion_matrix_ncs[1, 1])
    acc_cs = (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0] + confusion_matrix_cs[0, 1] +confusion_matrix_cs[1, 1])
    acc_cv = (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0] + confusion_matrix_cv[0, 1] +confusion_matrix_cv[1, 1])
    acc_cs_cv = (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    acc_conv1d = np.concatenate((acc_ncs, acc_cv, acc_cs, acc_cs_cv))


    spec_ncs = (confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1])
    spec_cs = (confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    spec_cv = (confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    spec_cs_cv = (confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    spec_conv1d = np.concatenate((spec_ncs, spec_cv, spec_cs, spec_cs_cv))


    sens_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    sens_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    sens_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[1, 0])
    sens_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    sens_conv1d = np.concatenate((sens_ncs, sens_cv, sens_cs, sens_cs_cv))


    recal_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    recal_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    recal_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])
    recal_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    recal_conv1d = np.concatenate((recal_ncs, recal_cv, recal_cs, recal_cs_cv))


    prec_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])
    prec_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])
    prec_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])
    prec_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])
    prec_conv1d = np.concatenate((prec_ncs, prec_cv, prec_cs, prec_cs_cv))


    f_ncs = 2 * recal_ncs * prec_ncs/ (recal_ncs + prec_ncs)
    f_cs = 2 * recal_cs * prec_cs / (recal_cs + prec_cs)
    f_cv = 2 * recal_cv * prec_cv / (recal_cv + prec_cv)
    f_cs_cv = 2 * recal_cs_cv * prec_cs_cv / (recal_cs_cv + prec_cs_cv)
    f_conv1d = np.concatenate((f_ncs, f_cv, f_cs, f_cs_cv))


    mcc_ncs = (confusion_matrix_ncs[0, 0] * confusion_matrix_ncs[1, 1] - confusion_matrix_ncs[0, 1] *confusion_matrix_ncs[1, 0])/np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])*(confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1])*(confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])*(confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_cs = (confusion_matrix_cs[0, 0] * confusion_matrix_cs[1, 1] - confusion_matrix_cs[0, 1] *confusion_matrix_cs[1, 0])/np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])*(confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1])*(confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])*(confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_cv = (confusion_matrix_cv[0, 0] * confusion_matrix_cv[1, 1] - confusion_matrix_cv[0, 1] *confusion_matrix_cv[1, 0])/np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])*(confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1])*(confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])*(confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_cs_cv = (confusion_matrix_cv_cs[0, 0] * confusion_matrix_cv_cs[1, 1] - confusion_matrix_cv_cs[0, 1] * confusion_matrix_cv_cs[1, 0])/np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])*(confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1])*(confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])*(confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_conv1d = np.concatenate((mcc_ncs, mcc_cv, mcc_cs, mcc_cs_cv))


# conv2d

    with open('conf_matrix_conv2d_ncs.pkl', 'rb') as f:
        confusion_matrix_ncs = pickle.load(f)

    with open('conf_matrix_conv2d_cs.pkl', 'rb') as f:
        confusion_matrix_cs = pickle.load(f)

    with open('conf_matrix_conv2d_cvs.pkl', 'rb') as f:
        confusion_matrix_cv = pickle.load(f)

    with open('conf_matrix_conv2d_cv_cs.pkl', 'rb') as f:
        confusion_matrix_cv_cs = pickle.load(f)

    # calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC

    acc_ncs = (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[0, 1] +confusion_matrix_ncs[1, 1])
    acc_cs = (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0] + confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    acc_cv = (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0] + confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    acc_cs_cv = (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[0, 1] +confusion_matrix_cv_cs[1, 1])
    acc_conv2d = np.concatenate((acc_ncs, acc_cv, acc_cs, acc_cs_cv))

    spec_ncs = (confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1])
    spec_cs = (confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    spec_cv = (confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    spec_cs_cv = (confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    spec_conv2d = np.concatenate((spec_ncs, spec_cv, spec_cs, spec_cs_cv))

    sens_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    sens_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    sens_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[1, 0])
    sens_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    sens_conv2d = np.concatenate((sens_ncs, sens_cv, sens_cs, sens_cs_cv))

    recal_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    recal_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    recal_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])
    recal_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    recal_conv2d = np.concatenate((recal_ncs, recal_cv, recal_cs, recal_cs_cv))

    prec_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])
    prec_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])
    prec_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])
    prec_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])
    prec_conv2d = np.concatenate((prec_ncs, prec_cv, prec_cs, prec_cs_cv))

    f_ncs = 2 * recal_ncs * prec_ncs / (recal_ncs + prec_ncs)
    f_cs = 2 * recal_cs * prec_cs / (recal_cs + prec_cs)
    f_cv = 2 * recal_cv * prec_cv / (recal_cv + prec_cv)
    f_cs_cv = 2 * recal_cs_cv * prec_cs_cv / (recal_cs_cv + prec_cs_cv)
    f_conv2d = np.concatenate((f_ncs, f_cv, f_cs, f_cs_cv))

    mcc_ncs = (confusion_matrix_ncs[0, 0] * confusion_matrix_ncs[1, 1] - confusion_matrix_ncs[0, 1] *confusion_matrix_ncs[1, 0]) / np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1]) * (confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1]) * (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0]) * (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_cs = (confusion_matrix_cs[0, 0] * confusion_matrix_cs[1, 1] - confusion_matrix_cs[0, 1] * confusion_matrix_cs[1, 0]) / np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1]) * (confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1]) * (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0]) * (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_cv = (confusion_matrix_cv[0, 0] * confusion_matrix_cv[1, 1] - confusion_matrix_cv[0, 1] * confusion_matrix_cv[1, 0]) / np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1]) * (confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1]) * (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0]) * (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_cs_cv = (confusion_matrix_cv_cs[0, 0] * confusion_matrix_cv_cs[1, 1] - confusion_matrix_cv_cs[0, 1] *confusion_matrix_cv_cs[1, 0]) / np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1]) * (confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1]) * (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0]) * (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_conv2d = np.concatenate((mcc_ncs, mcc_cv, mcc_cs, mcc_cs_cv))

# conv1dlstm

    with open('conf_matrix_conv1dlstm_uni_ncs.pkl', 'rb') as f:
        confusion_matrix_ncs = pickle.load(f)

    with open('conf_matrix_conv1dlstm_cs.pkl', 'rb') as f:
        confusion_matrix_cs = pickle.load(f)

    with open('conf_matrix_conv1dlstm_uni_cvs.pkl', 'rb') as f:
        confusion_matrix_cv = pickle.load(f)

    with open('conf_matrix_conv1dlstm_cv_cs.pkl', 'rb') as f:
        confusion_matrix_cv_cs = pickle.load(f)

    # calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC

    acc_ncs = (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[0, 1] +confusion_matrix_ncs[1, 1])
    acc_cs = (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0] + confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    acc_cv = (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0] + confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    acc_cs_cv = (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[0, 1] +confusion_matrix_cv_cs[1, 1])
    acc_conv1dlstm = np.concatenate((acc_ncs, acc_cv, acc_cs, acc_cs_cv))

    spec_ncs = (confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1])
    spec_cs = (confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    spec_cv = (confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    spec_cs_cv = (confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    spec_conv1dlstm = np.concatenate((spec_ncs, spec_cv, spec_cs, spec_cs_cv))

    sens_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    sens_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    sens_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[1, 0])
    sens_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    sens_conv1dlstm = np.concatenate((sens_ncs, sens_cv, sens_cs, sens_cs_cv))

    recal_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    recal_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    recal_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])
    recal_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    recal_conv1dlstm = np.concatenate((recal_ncs, recal_cv, recal_cs, recal_cs_cv))

    prec_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])
    prec_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])
    prec_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])
    prec_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])
    prec_conv1dlstm = np.concatenate((prec_ncs, prec_cv, prec_cs, prec_cs_cv))

    f_ncs = 2 * recal_ncs * prec_ncs / (recal_ncs + prec_ncs)
    f_cs = 2 * recal_cs * prec_cs / (recal_cs + prec_cs)
    f_cv = 2 * recal_cv * prec_cv / (recal_cv + prec_cv)
    f_cs_cv = 2 * recal_cs_cv * prec_cs_cv / (recal_cs_cv + prec_cs_cv)
    f_conv1dlstm = np.concatenate((f_ncs, f_cv, f_cs, f_cs_cv))

    mcc_ncs = (confusion_matrix_ncs[0, 0] * confusion_matrix_ncs[1, 1] - confusion_matrix_ncs[0, 1] *confusion_matrix_ncs[1, 0]) / np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1]) * (confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1]) * (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0]) * (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_cs = (confusion_matrix_cs[0, 0] * confusion_matrix_cs[1, 1] - confusion_matrix_cs[0, 1] * confusion_matrix_cs[1, 0]) / np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1]) * (confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1]) * (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0]) * (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_cv = (confusion_matrix_cv[0, 0] * confusion_matrix_cv[1, 1] - confusion_matrix_cv[0, 1] * confusion_matrix_cv[1, 0]) / np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1]) * (confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1]) * (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0]) * (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_cs_cv = (confusion_matrix_cv_cs[0, 0] * confusion_matrix_cv_cs[1, 1] - confusion_matrix_cv_cs[0, 1] *confusion_matrix_cv_cs[1, 0]) / np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1]) * (confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1]) * (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0]) * (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_conv1dlstm = np.concatenate((mcc_ncs, mcc_cv, mcc_cs, mcc_cs_cv))


# conv2dlstm

    with open('conf_matrix_conv2dlstm_uni_ncs.pkl', 'rb') as f:
        confusion_matrix_ncs = pickle.load(f)

    with open('conf_matrix_conv2dlstm_cs.pkl', 'rb') as f:
        confusion_matrix_cs = pickle.load(f)

    with open('conf_matrix_conv2dlstm_uni_cvs.pkl', 'rb') as f:
        confusion_matrix_cv = pickle.load(f)

    with open('conf_matrix_conv2dlstm_cv_cs.pkl', 'rb') as f:
        confusion_matrix_cv_cs = pickle.load(f)

    # calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC

    acc_ncs = (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[0, 1] +confusion_matrix_ncs[1, 1])
    acc_cs = (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0] + confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    acc_cv = (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0] + confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    acc_cs_cv = (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[0, 1] +confusion_matrix_cv_cs[1, 1])
    acc_conv2dlstm = np.concatenate((acc_ncs, acc_cv, acc_cs, acc_cs_cv))

    spec_ncs = (confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1])
    spec_cs = (confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    spec_cv = (confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    spec_cs_cv = (confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    spec_conv2dlstm = np.concatenate((spec_ncs, spec_cv, spec_cs, spec_cs_cv))

    sens_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    sens_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    sens_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[1, 0])
    sens_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    sens_conv2dlstm = np.concatenate((sens_ncs, sens_cv, sens_cs, sens_cs_cv))

    recal_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    recal_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    recal_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])
    recal_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    recal_conv2dlstm = np.concatenate((recal_ncs, recal_cv, recal_cs, recal_cs_cv))

    prec_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])
    prec_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])
    prec_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])
    prec_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])
    prec_conv2dlstm = np.concatenate((prec_ncs, prec_cv, prec_cs, prec_cs_cv))

    f_ncs = 2 * recal_ncs * prec_ncs / (recal_ncs + prec_ncs)
    f_cs = 2 * recal_cs * prec_cs / (recal_cs + prec_cs)
    f_cv = 2 * recal_cv * prec_cv / (recal_cv + prec_cv)
    f_cs_cv = 2 * recal_cs_cv * prec_cs_cv / (recal_cs_cv + prec_cs_cv)
    f_conv2dlstm = np.concatenate((f_ncs, f_cv, f_cs, f_cs_cv))

    mcc_ncs = (confusion_matrix_ncs[0, 0] * confusion_matrix_ncs[1, 1] - confusion_matrix_ncs[0, 1] *confusion_matrix_ncs[1, 0]) / np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1]) * (confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1]) * (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0]) * (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_cs = (confusion_matrix_cs[0, 0] * confusion_matrix_cs[1, 1] - confusion_matrix_cs[0, 1] * confusion_matrix_cs[1, 0]) / np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1]) * (confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1]) * (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0]) * (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_cv = (confusion_matrix_cv[0, 0] * confusion_matrix_cv[1, 1] - confusion_matrix_cv[0, 1] * confusion_matrix_cv[1, 0]) / np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1]) * (confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1]) * (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0]) * (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_cs_cv = (confusion_matrix_cv_cs[0, 0] * confusion_matrix_cv_cs[1, 1] - confusion_matrix_cv_cs[0, 1] *confusion_matrix_cv_cs[1, 0]) / np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1]) * (confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1]) * (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0]) * (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_conv2dlstm = np.concatenate((mcc_ncs, mcc_cv, mcc_cs, mcc_cs_cv))


# lstm

    with open('conf_matrix_lstm_unid_ncs.pkl', 'rb') as f:
        confusion_matrix_ncs = pickle.load(f)

    with open('conf_matrix_lstm_uni_cs.pkl', 'rb') as f:
        confusion_matrix_cs = pickle.load(f)

    with open('conf_matrix_lstm_uni_cvs.pkl', 'rb') as f:
        confusion_matrix_cv = pickle.load(f)

    with open('conf_matrix_lstm_cv_cs.pkl', 'rb') as f:
        confusion_matrix_cv_cs = pickle.load(f)

    # calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC

    acc_ncs = (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[0, 1] +confusion_matrix_ncs[1, 1])
    acc_cs = (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0] + confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    acc_cv = (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0] + confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    acc_cs_cv = (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[0, 1] +confusion_matrix_cv_cs[1, 1])
    acc_lstm = np.concatenate((acc_ncs, acc_cv, acc_cs, acc_cs_cv))

    spec_ncs = (confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1])
    spec_cs = (confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    spec_cv = (confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    spec_cs_cv = (confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    spec_lstm = np.concatenate((spec_ncs, spec_cv, spec_cs, spec_cs_cv))

    sens_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    sens_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    sens_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[1, 0])
    sens_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    sens_lstm = np.concatenate((sens_ncs, sens_cv, sens_cs, sens_cs_cv))

    recal_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    recal_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    recal_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])
    recal_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    recal_lstm = np.concatenate((recal_ncs, recal_cv, recal_cs, recal_cs_cv))

    prec_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])
    prec_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])
    prec_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])
    prec_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])
    prec_lstm = np.concatenate((prec_ncs, prec_cv, prec_cs, prec_cs_cv))

    f_ncs = 2 * recal_ncs * prec_ncs / (recal_ncs + prec_ncs)
    f_cs = 2 * recal_cs * prec_cs / (recal_cs + prec_cs)
    f_cv = 2 * recal_cv * prec_cv / (recal_cv + prec_cv)
    f_cs_cv = 2 * recal_cs_cv * prec_cs_cv / (recal_cs_cv + prec_cs_cv)
    f_lstm = np.concatenate((f_ncs, f_cv, f_cs, f_cs_cv))

    mcc_ncs = (confusion_matrix_ncs[0, 0] * confusion_matrix_ncs[1, 1] - confusion_matrix_ncs[0, 1] *confusion_matrix_ncs[1, 0]) / np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1]) * (confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1]) * (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0]) * (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_cs = (confusion_matrix_cs[0, 0] * confusion_matrix_cs[1, 1] - confusion_matrix_cs[0, 1] * confusion_matrix_cs[1, 0]) / np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1]) * (confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1]) * (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0]) * (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_cv = (confusion_matrix_cv[0, 0] * confusion_matrix_cv[1, 1] - confusion_matrix_cv[0, 1] * confusion_matrix_cv[1, 0]) / np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1]) * (confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1]) * (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0]) * (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_cs_cv = (confusion_matrix_cv_cs[0, 0] * confusion_matrix_cv_cs[1, 1] - confusion_matrix_cv_cs[0, 1] *confusion_matrix_cv_cs[1, 0]) / np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1]) * (confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1]) * (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0]) * (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_lstm = np.concatenate((mcc_ncs, mcc_cv, mcc_cs, mcc_cs_cv))


# bilstm

    with open('conf_matrix_lstm_ncs.pkl', 'rb') as f:
        confusion_matrix_ncs = pickle.load(f)

    with open('conf_matrix_lstm_bi_cs.pkl', 'rb') as f:
        confusion_matrix_cs = pickle.load(f)

    with open('conf_matrix_lstm_cvs.pkl', 'rb') as f:
        confusion_matrix_cv = pickle.load(f)

    with open('conf_matrix_lstm_bi_cv_cs.pkl', 'rb') as f:
        confusion_matrix_cv_cs = pickle.load(f)

    # calcolo la ACC, sensitività, specificità, recall, precision, misura F, la MCC

    acc_ncs = (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[0, 1] +confusion_matrix_ncs[1, 1])
    acc_cs = (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0] + confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    acc_cv = (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0] + confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    acc_cs_cv = (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[0, 1] +confusion_matrix_cv_cs[1, 1])
    acc_bilstm = np.concatenate((acc_ncs, acc_cv, acc_cs, acc_cs_cv))

    spec_ncs = (confusion_matrix_ncs[1, 1]) / (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1])
    spec_cs = (confusion_matrix_cs[1, 1]) / (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1])
    spec_cv = (confusion_matrix_cv[1, 1]) / (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1])
    spec_cs_cv = (confusion_matrix_cv_cs[1, 1]) / (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1])
    spec_bilstm = np.concatenate((spec_ncs, spec_cv, spec_cs, spec_cs_cv))

    sens_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    sens_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    sens_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[i][0, 0] + confusion_matrix_cv[1, 0])
    sens_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    sens_bilstm = np.concatenate((sens_ncs, sens_cv, sens_cs, sens_cs_cv))

    recal_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0])
    recal_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0])
    recal_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0])
    recal_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0])
    recal_bilstm = np.concatenate((recal_ncs, recal_cv, recal_cs, recal_cs_cv))

    prec_ncs = (confusion_matrix_ncs[0, 0]) / (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1])
    prec_cs = (confusion_matrix_cs[0, 0]) / (confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1])
    prec_cv = (confusion_matrix_cv[0, 0]) / (confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1])
    prec_cs_cv = (confusion_matrix_cv_cs[0, 0]) / (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1])
    prec_bilstm = np.concatenate((prec_ncs, prec_cv, prec_cs, prec_cs_cv))

    f_ncs = 2 * recal_ncs * prec_ncs / (recal_ncs + prec_ncs)
    f_cs = 2 * recal_cs * prec_cs / (recal_cs + prec_cs)
    f_cv = 2 * recal_cv * prec_cv / (recal_cv + prec_cv)
    f_cs_cv = 2 * recal_cs_cv * prec_cs_cv / (recal_cs_cv + prec_cs_cv)
    f_bilstm = np.concatenate((f_ncs, f_cv, f_cs, f_cs_cv))

    mcc_ncs = (confusion_matrix_ncs[0, 0] * confusion_matrix_ncs[1, 1] - confusion_matrix_ncs[0, 1] *confusion_matrix_ncs[1, 0]) / np.sqrt((confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[0, 1]) * (confusion_matrix_ncs[1, 0] + confusion_matrix_ncs[1, 1]) * (confusion_matrix_ncs[0, 0] + confusion_matrix_ncs[1, 0]) * (confusion_matrix_ncs[0, 1] + confusion_matrix_ncs[1, 1]))
    mcc_cs = (confusion_matrix_cs[0, 0] * confusion_matrix_cs[1, 1] - confusion_matrix_cs[0, 1] * confusion_matrix_cs[1, 0]) / np.sqrt((confusion_matrix_cs[0, 0] + confusion_matrix_cs[0, 1]) * (confusion_matrix_cs[1, 0] + confusion_matrix_cs[1, 1]) * (confusion_matrix_cs[0, 0] + confusion_matrix_cs[1, 0]) * (confusion_matrix_cs[0, 1] + confusion_matrix_cs[1, 1]))
    mcc_cv = (confusion_matrix_cv[0, 0] * confusion_matrix_cv[1, 1] - confusion_matrix_cv[0, 1] * confusion_matrix_cv[1, 0]) / np.sqrt((confusion_matrix_cv[0, 0] + confusion_matrix_cv[0, 1]) * (confusion_matrix_cv[1, 0] + confusion_matrix_cv[1, 1]) * (confusion_matrix_cv[0, 0] + confusion_matrix_cv[1, 0]) * (confusion_matrix_cv[0, 1] + confusion_matrix_cv[1, 1]))
    mcc_cs_cv = (confusion_matrix_cv_cs[0, 0] * confusion_matrix_cv_cs[1, 1] - confusion_matrix_cv_cs[0, 1] *confusion_matrix_cv_cs[1, 0]) / np.sqrt((confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[0, 1]) * (confusion_matrix_cv_cs[1, 0] + confusion_matrix_cv_cs[1, 1]) * (confusion_matrix_cv_cs[0, 0] + confusion_matrix_cv_cs[1, 0]) * (confusion_matrix_cv_cs[0, 1] + confusion_matrix_cv_cs[1, 1]))
    mcc_bilstm = np.concatenate((mcc_ncs, mcc_cv, mcc_cs, mcc_cs_cv))