import pickle
import numpy as np

# analisi statistica ML

with open('confusion_matrix_ncs_mc.pkl', 'rb') as f:
    conf_matrix_ncs = pickle.load(f)

with open('confusion_matrix_cs_mc.pkl', 'rb') as f:
    conf_matrix_cs = pickle.load(f)

with open('confusion_matrix_cv_mc.pkl', 'rb') as f:
    conf_matrix_cv = pickle.load(f)

with open('Confusion_matrix_cs_cv_mc.pkl', 'rb') as f:
    conf_matrix_cs_cv = pickle.load(f)


class calcolo_statistiche:
    def __init__(self, conf_matrix):
        self.conf_matrix = conf_matrix


    def specificita(self):

        specificita_list = []

        for j in range(15):
            TP = self.conf_matrix[j, j]
            FN = np.sum(self.conf_matrix[j, :]) - TP
            FP = np.sum(self.conf_matrix[:, j]) - TP
            TN = np.sum(self.conf_matrix) - TP - FN - FP

            specificita = TN / (FP + TN)
            specificita_list.append(specificita)

        specificita_media = sum(specificita_list) / len(specificita_list)

        return specificita_media

    def recall(self):

        recall_list = []

        for j in range(15):
            TP = self.conf_matrix[j, j]
            FN = np.sum(self.conf_matrix[j, :]) - TP
            FP = np.sum(self.conf_matrix[:, j]) - TP
            TN = np.sum(self.conf_matrix) - TP - FN - FP

            recall = TP / (TP + FN)
            recall_list.append(recall)

        recall_media = sum(recall_list) / len(recall_list)

        return recall_media


    def precision(self):

        precision_list = []

        for j in range(15):
            TP = self.conf_matrix[j, j]
            FN = np.sum(self.conf_matrix[j, :]) - TP
            FP = np.sum(self.conf_matrix[:, j]) - TP
            TN = np.sum(self.conf_matrix) - TP - FN - FP

            precision = TP / (FP + TP)
            precision_list.append(precision)

        precision_media = sum(precision_list) / len(precision_list)

        return precision_media


    def misura_mcc(self):

        mcc_list = []

        for j in range(15):
            TP = self.conf_matrix[j, j]
            FN = np.sum(self.conf_matrix[j, :]) - TP
            FP = np.sum(self.conf_matrix[:, j]) - TP
            TN = np.sum(self.conf_matrix) - TP - FN - FP


            mcc = (TP*TN - FP*FN)/np.sqrt((TP + FP)*(FN + TN)*(TP + FN)*(FP + TN))
            mcc_list.append(mcc)

        mcc_media = sum(mcc_list) / len(mcc_list)

        return mcc_media



spec_list_ncs = []
spec_list_cs = []
spec_list_cv = []
spec_list_cs_cv = []

for i in range(5):
    spec_ncs = calcolo_statistiche(conf_matrix_ncs[i])
    spec_list_ncs.append(spec_ncs.specificita())

    spec_cs = calcolo_statistiche(conf_matrix_cs[i])
    spec_list_cs.append(spec_cs.specificita())

    spec_cv = calcolo_statistiche(conf_matrix_cv[i])
    spec_list_cv.append(spec_cv.specificita())

    spec_cs_cv = calcolo_statistiche(conf_matrix_cs_cv[i])
    spec_list_cs_cv.append(spec_cs_cv.specificita())



print("spec ncs", spec_list_ncs)
print("spec cs", spec_list_cs)
print("spec cv", spec_list_cv)
print("spec cs_cv", spec_list_cs_cv)


recall_list_ncs = []
recall_list_cs = []
recall_list_cv = []
recall_list_cs_cv = []

for i in range(5):
    recall_ncs = calcolo_statistiche(conf_matrix_ncs[i])
    recall_list_ncs.append(recall_ncs.recall())

    recall_cs = calcolo_statistiche(conf_matrix_cs[i])
    recall_list_cs.append(recall_cs.recall())

    recall_cv = calcolo_statistiche(conf_matrix_cv[i])
    recall_list_cv.append(recall_cv.recall())

    recall_cs_cv = calcolo_statistiche(conf_matrix_cs_cv[i])
    recall_list_cs_cv.append(recall_cs_cv.recall())


print("rec ncs", recall_list_ncs)
print("rec cs", recall_list_cs)
print("rec cv", recall_list_cv)
print("rec cs_cv", recall_list_cs_cv)


precision_list_ncs = []
precision_list_cs = []
precision_list_cv = []
precision_list_cs_cv = []

for i in range(5):
    prec_ncs = calcolo_statistiche(conf_matrix_ncs[i])
    precision_list_ncs.append(prec_ncs.precision())

    prec_cs = calcolo_statistiche(conf_matrix_cs[i])
    precision_list_cs.append(prec_cs.precision())

    prec_cv = calcolo_statistiche(conf_matrix_cv[i])
    precision_list_cv.append(prec_cv.precision())

    prec_cs_cv = calcolo_statistiche(conf_matrix_cs_cv[i])
    precision_list_cs_cv.append(prec_cs_cv.precision())



print("prec ncs", precision_list_ncs)
print("prec cs", precision_list_cs)
print("prec cv", precision_list_cv)
print("prec cs_cv", precision_list_cs_cv)


mcc_list_ncs = []
mcc_list_cs = []
mcc_list_cv = []
mcc_list_cs_cv = []

for i in range(5):
    mcc_ncs = calcolo_statistiche(conf_matrix_ncs[i])
    mcc_list_ncs.append(mcc_ncs.misura_mcc())

    mcc_cs = calcolo_statistiche(conf_matrix_cs[i])
    mcc_list_cs.append(mcc_cs.misura_mcc())

    mcc_cv = calcolo_statistiche(conf_matrix_cv[i])
    mcc_list_cv.append(mcc_cv.misura_mcc())

    mcc_cs_cv = calcolo_statistiche(conf_matrix_cs_cv[i])
    mcc_list_cs_cv.append(mcc_cs_cv.misura_mcc())


print("mcc ncs", mcc_list_ncs)
print("mcc cs", mcc_list_cs)
print("mcc cv", mcc_list_cv)
print("mcc cs_cv", mcc_list_cs_cv)

# analisi statistica DL

with open('conf_matrix_conv1d_ncs_mc.pkl','rb') as f:
    conf_matrix_ncs = pickle.load(f)

with open('conf_matrix_conv1d_cs_mc.pkl','rb') as f:
    conf_matrix_cs = pickle.load(f)

with open('conf_matrix_conv1d_cv_mc.pkl','rb') as f:
    conf_matrix_cv = pickle.load(f)

with open('conf_matrix_conv1d_cv_cs_mc.pkl','rb') as f:
    conf_matrix_cs_cv = pickle.load(f)


# conv1d


values = calcolo_statistiche(conf_matrix_ncs)
spec_ncs = values.specificita()
values = calcolo_statistiche(conf_matrix_cs)
spec_cs = values.specificita()
values = calcolo_statistiche(conf_matrix_cv)
spec_cv = values.specificita()
values = calcolo_statistiche(conf_matrix_cs_cv)
spec_cs_cv = values.specificita()


print("spec conv1d ncs", spec_ncs)
print("spec conv1d cs", spec_cs)
print("spec conv1d cv", spec_cv)
print("spec conv1d cs_cv", spec_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
recall_ncs = values.recall()
values = calcolo_statistiche(conf_matrix_cs)
recall_cs = values.recall()
values = calcolo_statistiche(conf_matrix_cv)
recall_cv = values.recall()
values = calcolo_statistiche(conf_matrix_cs_cv)
recall_cs_cv = values.recall()


print("rec conv1d ncs", recall_ncs)
print("rec conv1d cs", recall_cs)
print("rec conv1d cv", recall_cv)
print("rec conv1d cs_cv", recall_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
prec_ncs = values.precision()
values = calcolo_statistiche(conf_matrix_cs)
prec_cs = values.precision()
values = calcolo_statistiche(conf_matrix_cv)
prec_cv = values.precision()
values = calcolo_statistiche(conf_matrix_cs_cv)
prec_cs_cv = values.precision()

print("prec conv1d ncs", prec_ncs)
print("prec conv1d cs", prec_cs)
print("prec conv1d cv", prec_cv)
print("prec conv1d cs_cv", prec_cs_cv)



values = calcolo_statistiche(conf_matrix_ncs)
mcc_ncs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs)
mcc_cs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cv)
mcc_cv = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs_cv)
mcc_cs_cv = values.misura_mcc()

print("mcc conv1d ncs", mcc_ncs)
print("mcc conv1d cs", mcc_cs)
print("mcc conv1d cv", mcc_cv)
print("mcc conv1d cs_cv", mcc_cs_cv)

# conv2d

with open('conf_matrix_conv2d_ncs_mc.pkl','rb') as f:
    conf_matrix_ncs = pickle.load(f)

with open('conf_matrix_conv2d_cs_mc.pkl','rb') as f:
    conf_matrix_cs = pickle.load(f)

with open('conf_matrix_conv2d_cv_mc.pkl','rb') as f:
    conf_matrix_cv = pickle.load(f)

with open('conf_matrix_conv2d_cs_cv_mc.pkl','rb') as f:
    conf_matrix_cs_cv = pickle.load(f)



values = calcolo_statistiche(conf_matrix_ncs)
spec_ncs = values.specificita()
values = calcolo_statistiche(conf_matrix_cs)
spec_cs = values.specificita()
values = calcolo_statistiche(conf_matrix_cv)
spec_cv = values.specificita()
values = calcolo_statistiche(conf_matrix_cs_cv)
spec_cs_cv = values.specificita()

print("spec conv2d ncs", spec_ncs)
print("spec conv2d cs", spec_cs)
print("spec conv2d cv", spec_cv)
print("spec conv2d cs_cv", spec_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
recall_ncs = values.recall()
values = calcolo_statistiche(conf_matrix_cs)
recall_cs = values.recall()
values = calcolo_statistiche(conf_matrix_cv)
recall_cv = values.recall()
values = calcolo_statistiche(conf_matrix_cs_cv)
recall_cs_cv = values.recall()


print("rec conv2d ncs", recall_ncs)
print("rec conv2d cs", recall_cs)
print("rec conv2d cv", recall_cv)
print("rec conv2d cs_cv", recall_cs_cv)




values = calcolo_statistiche(conf_matrix_ncs)
prec_ncs = values.precision()
values = calcolo_statistiche(conf_matrix_cs)
prec_cs = values.precision()
values = calcolo_statistiche(conf_matrix_cv)
prec_cv = values.precision()
values = calcolo_statistiche(conf_matrix_cs_cv)
prec_cs_cv = values.precision()

print("prec conv2d ncs", prec_ncs)
print("prec conv2d cs", prec_cs)
print("prec conv2d cv", prec_cv)
print("prec conv2d cs_cv", prec_cs_cv)



values = calcolo_statistiche(conf_matrix_ncs)
mcc_ncs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs)
mcc_cs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cv)
mcc_cv = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs_cv)
mcc_cs_cv = values.misura_mcc()


print("mcc conv2d ncs", mcc_ncs)
print("mcc conv2d cs", mcc_cs)
print("mcc conv2d cv", mcc_cv)
print("ncc conv2d cs_cv", mcc_cs_cv)


# conv1dlstm


with open('conf_matrix_conv1dlstm_ncs_mc.pkl','rb') as f:
    conf_matrix_ncs = pickle.load(f)

with open('conf_matrix_conv1dlstm_cs_mc.pkl','rb') as f:
    conf_matrix_cs = pickle.load(f)

with open('conf_matrix_conv1dlstm_cv_mc.pkl','rb') as f:
    conf_matrix_cv = pickle.load(f)

with open('conf_matrix_conv1dlstm_cs_cv_mc.pkl','rb') as f:
    conf_matrix_cs_cv = pickle.load(f)



values = calcolo_statistiche(conf_matrix_ncs)
spec_ncs = values.specificita()
values = calcolo_statistiche(conf_matrix_cs)
spec_cs = values.specificita()
values = calcolo_statistiche(conf_matrix_cv)
spec_cv = values.specificita()
values = calcolo_statistiche(conf_matrix_cs_cv)
spec_cs_cv = values.specificita()


print("spec conv1dlstm ncs", spec_ncs)
print("spec conv1dlstm cs", spec_cs)
print("spec conv1dlstm cv", spec_cv)
print("spec conv1dlstm cs_cv", spec_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
recall_ncs = values.recall()
values = calcolo_statistiche(conf_matrix_cs)
recall_cs = values.recall()
values = calcolo_statistiche(conf_matrix_cv)
recall_cv = values.recall()
values = calcolo_statistiche(conf_matrix_cs_cv)
recall_cs_cv = values.recall()


print("rec conv1dlstm ncs", recall_ncs)
print("rec conv1dlstm cs", recall_cs)
print("rec conv1dlstm cv", recall_cv)
print("rec conv1dlstm cs_cv", recall_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
prec_ncs = values.precision()
values = calcolo_statistiche(conf_matrix_cs)
prec_cs = values.precision()
values = calcolo_statistiche(conf_matrix_cv)
prec_cv = values.precision()
values = calcolo_statistiche(conf_matrix_cs_cv)
prec_cs_cv = values.precision()


print("prec conv1dlstm ncs", prec_ncs)
print("prec conv1dlstm cs", prec_cs)
print("prec conv1dlstm cv", prec_cv)
print("prec conv1dlstm cs_cv", prec_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
mcc_ncs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs)
mcc_cs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cv)
mcc_cv = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs_cv)
mcc_cs_cv = values.misura_mcc()


print("mcc conv1dlstm ncs", mcc_ncs)
print("mcc conv1dlstm cs", mcc_cs)
print("mcc conv1dlstm cv", mcc_cv)
print("mcc conv1dlstm cs_cv", mcc_cs_cv)



# conv2dlstm


with open('conf_matrix_conv2dlstm_ncs_mc.pkl','rb') as f:
    conf_matrix_ncs = pickle.load(f)

with open('conf_matrix_conv2dlstm_cs_mc.pkl','rb') as f:
    conf_matrix = pickle.load(f)

with open('conf_matrix_conv2dlstm_cs_mc1.pkl', 'rb') as f:
    conf_matrix1 = pickle.load(f)

conf_matrix_cs = np.round((conf_matrix + conf_matrix1)/2)

with open('conf_matrix_conv2dlstm_cv_mc.pkl','rb') as f:
    conf_matrix_cv = pickle.load(f)

with open('conf_matrix_conv2dlstm_cs_cv_mc.pkl','rb') as f:
    conf_matrix = pickle.load(f)

with open('conf_matrix_conv2dlstm_cs_cv_mc1.pkl', 'rb') as f:
    conf_matrix1 = pickle.load(f)

conf_matrix_cs_cv = np.round((conf_matrix + conf_matrix1)/2)


values = calcolo_statistiche(conf_matrix_ncs)
spec_ncs = values.specificita()
values = calcolo_statistiche(conf_matrix_cs)
spec_cs = values.specificita()
values = calcolo_statistiche(conf_matrix_cv)
spec_cv = values.specificita()
values = calcolo_statistiche(conf_matrix_cs_cv)
spec_cs_cv = values.specificita()

print("spec conv2dlstm ncs", spec_ncs)
print("spec conv2dlstm cs", spec_cs)
print("spec conv2dlstm cv", spec_cv)
print("spec conv2dlstm cs_cv", spec_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
recall_ncs = values.recall()
values = calcolo_statistiche(conf_matrix_cs)
recall_cs = values.recall()
values = calcolo_statistiche(conf_matrix_cv)
recall_cv = values.recall()
values = calcolo_statistiche(conf_matrix_cs_cv)
recall_cs_cv = values.recall()


print("rec conv2dlstm ncs", recall_ncs)
print("rec conv2dlstm cs", recall_cs)
print("rec conv2dlstm cv", recall_cv)
print("rec conv2dlstm cs_cv", recall_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
prec_ncs = values.precision()
values = calcolo_statistiche(conf_matrix_cs)
prec_cs = values.precision()
values = calcolo_statistiche(conf_matrix_cv)
prec_cv = values.precision()
values = calcolo_statistiche(conf_matrix_cs_cv)
prec_cs_cv = values.precision()


print("prec conv1dlstm ncs", prec_ncs)
print("prec conv1dlstm cs", prec_cs)
print("prec conv1dlstm cv", prec_cv)
print("prec conv1dlstm cs_cv", prec_cs_cv)


values = calcolo_statistiche(conf_matrix_ncs)
mcc_ncs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs)
mcc_cs = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cv)
mcc_cv = values.misura_mcc()
values = calcolo_statistiche(conf_matrix_cs_cv)
mcc_cs_cv = values.misura_mcc()

print("mcc conv2dlstm ncs", mcc_ncs)
print("mcc conv2dlstm cs", mcc_cs)
print("mcc conv2dlstm cv", mcc_cv)
print("mcc conv2dlstm cs_cv", mcc_cs_cv)

