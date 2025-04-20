import os
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


data_fold = "C:/Users/Giovanni Maria/Desktop/raw_npy"
extension = ".skeleton.npy"
data_list = [] # inizializzo una lista vuota
file_list = []

classi = ["A007","A008","A009","A027", "A042", "A043", "A046", "A047", "A054", "A059", "A060", "A069", "A070", "A080", "A099"]

for classe in classi:
    for filename in os.listdir(data_fold): # os.listdir mi ritorna la lista dei file della cartella adesso definisco la condizione
        if classe in filename:
            file_path = os.path.join(data_fold, filename) # costruisce il percorso completo del file
            try:
                data = np.load(file_path, allow_pickle = True).item() # np.load serve per leggere dati binari specifici (in formato npy), senza item il risultato di np.load sarebbe 0, cioè contiene il dizionario come unico elemento, permette di estarre quel singolo elemento dall'array
                data_list.append(data)
                file_list.append(filename)
            #print(f"Dati caricati da {filename}:")
            #print(data)
            except Exception as e:
                print(f"Errore durante l'apertura del file {filename}:{e}")


# determino il numero di volte in cui sono presenti le diverse classi

cont = 0
lista_num_classi = []

for classe in classi:
    for filename in file_list:
        if classe in filename:
            cont += 1

    lista_num_classi.append(cont)
    cont = 0

#print(f"numero ripetizioni classi{classi}",lista_num_classi)

# determino la lunghezza media della sequenza di ogni classe

lungh_media_seq_classe = []
lungh_seq_classe = []
for classe in classi:
    for data in data_list:
        if classe in data['file_name']:
            lunghezza = len(data['skel_body0'])
            lungh_seq_classe.append(lunghezza)
    media_lunghezza = np.mean(lungh_seq_classe)
    lungh_media_seq_classe.append(media_lunghezza)

    lungh_seq_classe = []

#print(f"lunghezza media sequenze classe {classi}",lungh_media_seq_classe)

# determino i pazienti per ogni classe

def estrazione_paz(s):
    inizio = s.index("P")
    fine = inizio + 4

    return s[inizio:fine]

pazienti = []
for data in data_list:
    substring = estrazione_paz(data['file_name'])
    pazienti.append(substring)

lista_pazienti = sorted(set(pazienti))

lista_cont_classe = []
lista_paz_classe = []

def num_paz_classe(paziente):

    lista_cont_classe = []
    cont = 0
    for classe in classi:
        for data in data_list:
            if paziente in data['file_name']:
                if classe in data['file_name']:
                    cont += 1
        lista_cont_classe.append(cont)
        cont = 0
    return lista_cont_classe


for paziente in lista_pazienti:
    paz_classe = num_paz_classe(paziente)
    #print(f"numero paziente {paziente} nelle classi {classi}",paz_classe)

# lunghezza media sequenze per paziente

def lunghezza_seq_media_paz_classe(paziente):

    media_lungh_seq_classe = []
    lungh_seq_classe = []
    for classe in classi:
        for data in data_list:
            if paziente in data['file_name']:
                if classe in data['file_name']:
                    lunghezza = len(data['skel_body0'])
                    lungh_seq_classe.append(lunghezza)

        media = np.mean(lungh_seq_classe)
        media_lungh_seq_classe.append(media)
        lungh_seq_classe = []

    return media_lungh_seq_classe


for paziente in lista_pazienti:
    len_seq_media_paz = lunghezza_seq_media_paz_classe(paziente)
    #print(f"lunghezza sequenze classe {classi} per paziente {paziente}",len_seq_media_paz)


# adesso considero le diverse camere, ovvero le viste frontali di camera 2 e 3, ovvero cam 2 e rip 2 e cam 3 rip 1

# numero dati nelle 2 viste frontali e numero dati nelle 2 viste frontali nelle diverse classi

cont_3_1 = 0
cont_2_2 = 0

camere = ["C002","C003"]
ripetizioni = ["R002","R001"]

for data in data_list:
    if "C002" in data['file_name']:
        if "R002" in data['file_name']:
            cont_2_2 += 1

for data in data_list:
    if "C003" in data['file_name']:
        if "R001" in data['file_name']:
            cont_3_1 += 1

cont = [cont_2_2, cont_3_1]
#print(f"numero ripetizioni in vista frontale camere {camere}",cont)


def numero_viste_frontali_classi(camera,classi,ripetizione):

    num_classi_vf = []
    cont = 0
    for classe in classi:
        for data in data_list:
            if classe in data['file_name']:
                if camera in data['file_name']:
                    if ripetizione in data['file_name']:
                        cont += 1

        num_classi_vf.append(cont)
        cont = 0

    return num_classi_vf


num_vf2_classi = numero_viste_frontali_classi("C002",classi,"R002")
#print(f"numero ripetizioni vf C002 in classi {classi}",num_vf2_classi,sum(num_vf2_classi))

num_vf3_classi = numero_viste_frontali_classi("C003",classi,"R001")
#print(f"numero ripetizioni vf C003 in classi {classi}",num_vf3_classi,sum(num_vf3_classi))
# le classi sono bilanciate nelle due viste

# lunghezza media sequenze nelle diverse classi riprese dalla camera 2 e 3 nelle viste frontali


def lunghezza_seq_media_vf(camera,ripetizione):

    lungh_list = []
    lungh_media_classe_list = []
    for classe in classi:
        for data in data_list:
            if camera in data['file_name']:
                if ripetizione in data['file_name']:
                    if classe in data['file_name']:
                        lunghezza = len(data['skel_body0'])
                        lungh_list.append(lunghezza)

        media_lungh = np.mean(lungh_list)
        lungh_media_classe_list.append(media_lungh)
        lungh_list = []

    return lungh_media_classe_list


lunghezza_seq_media_vf2 = lunghezza_seq_media_vf("C002","R002")
#print(f"lunghezza media sequenze in vf C002 in classi {classi}",lunghezza_seq_media_vf2)

lunghezza_seq_media_vf3 = lunghezza_seq_media_vf("C003","R001")
#print(f"lunghezza media sequenze in vf C003 in classi {classi}",lunghezza_seq_media_vf3)

# adesso faccio un'analisi cross view (setting)

# determino il numero di setting

def estrazione_paz(s):
    inizio = s.index("S")
    fine = inizio + 4
    return s[inizio:fine]

setting = []
for data in data_list:
    substring = estrazione_paz(data['file_name'])
    setting.append(substring)

lista_set = sorted(set(setting))

cont = 0
cont_lista_setting = []
for setting in lista_set:
    for data in data_list:
        if setting in data['file_name']:
            cont += 1
    cont_lista_setting.append(cont)
    cont = 0

i = 0
for i in range(len(cont_lista_setting)):
    print(f"numero ripetizioni setting {lista_set[i]}",cont_lista_setting[i])

# determino il numero di setting per classe

def numero_setting_classe(setting):

    lista_cont_setting_classe = []
    cont = 0
    for classe in classi:
        for data in data_list:
            if setting in data['file_name']:
                if classe in data['file_name']:
                    cont += 1

        lista_cont_setting_classe.append(cont)
        cont = 0

    return lista_cont_setting_classe


num_setting_classe = []
for setting in lista_set:
    num_setting_classe = numero_setting_classe(setting)
    #print(f"numero ripetizioni setting {setting} per classi {classi}",num_setting_classe,sum(num_setting_classe))


# determino il numero di setting per camera 2 e 3 vista frontale

def numero_setting_camera(setting,camera,ripetizione):


    cont = 0
    for data in data_list:
        if camera in data['file_name']:
            if ripetizione in data['file_name']:
                if setting in data['file_name']:
                        cont += 1

    return cont

numero_setting_cam2 = []
for setting in lista_set:
    num_set = numero_setting_camera(setting,"C002","R002")
    numero_setting_cam2.append(num_set)

numero_setting_cam3 = []
for setting in lista_set:
    num_set = numero_setting_camera(setting, "C003", "R001")
    numero_setting_cam3.append(num_set)

#print(f"numero setting {lista_set} in camera C002",numero_setting_cam2)
#print(f"numero setting {lista_set} in camera C003",numero_setting_cam3)


# numero setting per paziente

def numero_pazienti_setting(paziente):

    cont = 0
    num_set_paz = []
    for setting in lista_set:
        for data in data_list:
            if paziente in data['file_name']:
                if setting in data['file_name']:
                    cont += 1

        num_set_paz.append(cont)
        cont = 0

    return num_set_paz


num_paz_setting = []
for paziente in lista_pazienti:
    num_paz_setting = numero_pazienti_setting(paziente)
    #print(f"numero setting paziente {paziente}",num_paz_setting)


# numero setting in vista frontale camera 2 e 3 nelle diverse classi

def numero_setting_classe_vf(setting,camera,ripetizione):

    cont = 0
    num_sett_classe_vf = []
    for classe in classi:
        for data in data_list:
            if setting in data['file_name']:
                if camera in data['file_name']:
                    if ripetizione in data['file_name']:
                        if classe in data['file_name']:
                            cont += 1

        num_sett_classe_vf.append(cont)
        cont = 0

    return num_sett_classe_vf


num_set_vf2 = []
for setting in lista_set:
    num_setting_classe_vf2 = numero_setting_classe_vf(setting,"C002","R002")
    #print(f"numero setting {setting} nelle classi {classi} in vf2",num_setting_classe_vf2)
    num_set_vf2.append(num_setting_classe_vf2)



for setting in lista_set:
    num_setting_classe_vf3 = numero_setting_classe_vf(setting,"C003","R001")
    #print(f"numero setting {setting} nelle classi {classi} in vf3", num_setting_classe_vf3)


# numero setting per paziente in vf 2 e 3

def numero_pazienti_setting_vf(paziente,camera,ripetizione):

    num_setting = []
    cont = 0
    for setting in lista_set:
        for data in data_list:
            if camera in data['file_name']:
                if ripetizione in data['file_name']:
                    if paziente in data['file_name']:
                        if setting in data['file_name']:
                            cont += 1

        num_setting.append(cont)
        cont = 0

    return num_setting


for paziente in lista_pazienti:
    numero_setting_paz_vf2 = numero_pazienti_setting_vf(paziente,"C002","R002")
    print(f"numero setting per paziente {paziente} in vf2",numero_setting_paz_vf2)

for paziente in lista_pazienti:
    numero_setting_paz_vf3 = numero_pazienti_setting_vf(paziente,"C003","R001")
    #print(f"numero setting per paziente {paziente} in vf3", numero_setting_paz_vf3)


# lunghezza media sequenze setting (indipendentemente dalla classe o vf)


def lungh_media_seq_set(setting):

    lungh_set = []
    for data in data_list:
        if setting in data['file_name']:
            lunghezza = len(data['skel_body0'])
            lungh_set.append(lunghezza)

    return np.mean(lungh_set)

lungh_media_set = []
for setting in lista_set:
    lungh_med_set = lungh_media_seq_set(setting)
#    print(f"lungh media set {setting}",lungh_med_set)
    lungh_media_set.append(lungh_med_set)

# determino la lunghezza media set in vf 2 e 3

lungh_set = []
lungh_med_set_vf2 = []
for setting in lista_set:
    for data in data_list:
        if setting in data['file_name']:
            if "C002" in data['file_name']:
                if "R002" in data['file_name']:
                    lungh = len(data['skel_body0'])
                    lungh_set.append(lungh)

    media = np.mean(lungh_set)
    lungh_med_set_vf2.append(media)
    lungh_set = []


lungh_set = []
lungh_med_set_vf3 = []
for setting in lista_set:
    for data in data_list:
        if setting in data['file_name']:
            if "C003" in data['file_name']:
                if "R001" in data['file_name']:
                    lungh = len(data['skel_body0'])
                    lungh_set.append(lungh)

    media = np.mean(lungh_set)
    lungh_med_set_vf3.append(media)
    lungh_set = []

# determino la lunghezza media setting per classe

def lungh_media_seq_set_classe(setting):

    lungh_media_classe_set = []
    lungh_classe_set = []
    for classe in classi:
        for data in data_list:
            if setting in data['file_name']:
                if classe in data['file_name']:
                    lunghezza = len(data['skel_body0'])
                    lungh_classe_set.append(lunghezza)

        media = np.mean(lungh_classe_set)
        lungh_media_classe_set.append(media)
        lungh_classe_set = []

    return lungh_media_classe_set


for setting in lista_set:
    lungh_med_set_classe = lungh_media_seq_set_classe(setting)
    #print(f"lungh media set {setting} per classi {classi}",lungh_med_set_classe)

# lunghezza media seq setting per classi in vf 2 e 3

def lungh_media_seq_set_classe_vf(setting,camera,ripetizione):

    lungh_media_classe_set = []
    lungh_classe = []
    for classe in classi:
        for data in data_list:
            if camera in data['file_name']:
                if ripetizione in data['file_name']:
                    if setting in data['file_name']:
                        if classe in data['file_name']:
                            lunghezza = len(data['skel_body0'])
                            lungh_classe.append(lunghezza)

        media = np.mean(lungh_classe)
        lungh_media_classe_set.append(media)
        lungh_classe = []

    return lungh_media_classe_set


for setting in lista_set:
    lungh_med_set_classe_vf2 = lungh_media_seq_set_classe_vf(setting,"C002","R002")
    print(f"lunghezza media set {setting} in classi {classi} in vf2",lungh_med_set_classe_vf2)


for setting in lista_set:
    lungh_med_set_classe_vf3 = lungh_media_seq_set_classe_vf(setting, "C003", "R001")
    #print(f"lunghezza media set {setting} in classi {classi} in vf3", lungh_med_set_classe_vf3)



######################################################

def lungh_media_seq_paz_vf(classe, paziente):

    lungh_paz = []

    for data in data_list:
        if classe in data['file_name']:
            if "C002" in data['file_name']:
                if "R002" in data['file_name']:
                    if paziente in data['file_name']:
                        lunghezza = len(data['skel_body0'])
                        lungh_paz.append(lunghezza)

    media = np.mean(lungh_paz)

    return media


lungh_paz_classe27 = []
for paziente in lista_pazienti:
    paz_classe =  lungh_media_seq_paz_vf("A027",paziente)
    lungh_paz_classe27.append(paz_classe)

lungh_paz_classe60 = []
for paziente in lista_pazienti:
    paz_classe =  lungh_media_seq_paz_vf("A060",paziente)
    lungh_paz_classe60.append(paz_classe)

lungh_paz_classe80 = []
for paziente in lista_pazienti:
    paz_classe =  lungh_media_seq_paz_vf("A080",paziente)
    lungh_paz_classe80.append(paz_classe)



keys_paz = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32',
            '33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60',
            '61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90',
            '91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106']


figure, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

ax1.bar(keys_paz, lungh_paz_classe27)
ax1.set_ylabel('jump up')

ax2.bar(keys_paz, lungh_paz_classe60)
ax2.set_ylabel('walking apart')

ax3.bar(keys_paz, lungh_paz_classe80)
ax3.set_ylabel('squat down')

plt.show()

#################################################


# numero setting in vf2

cont = 0
freq_set_vf2 = []
for setting in lista_set:
    for data in data_list:
        if setting in data['file_name']:
            if "C002" in data['file_name']:
                if "R002" in data['file_name']:
                    cont += 1

    freq_set_vf2.append(cont)
    cont = 0

print(freq_set_vf2)


## analisi qualitativa

# diagramma a torta per vedere se c'è un bilanciamento tra le due classi

plt.style.use("ggplot")
labels = ['throw','sit down','stand up',' jump up','staggering','falling down', 'touch back','touch neck','point finger','walking towards',' walking apart','thumb up','thumb down',' squat down','run on the spot']
colors = ['red','blue','green','yellow','purple','orange','cyan','magenta','brown','pink','gray','olive','lime','navy','gold']
wedges, texts, autotexts = plt.pie(lista_num_classi, labels = labels, autopct = lambda p: '{:.0f}'.format(p * sum(lista_num_classi)/100), startangle=90, colors = colors)
#plt.pie(slices, labels = labels, autopct = lambda x: f'{x:.1f}%')
plt.title("Proporzione dei pazienti nelle diverse classi")

for text in texts:
    if text.get_text() == "touch neck":
        pos = text.get_position()
        text.set_position((pos[0] + 0.2, pos[1]))

plt.show()

## istogramma per vedere la lunghezza medi delle sequenze delle classi

classi = ['A7','A8','A9','A27','A42','A43','A46','A47','A54','A59','A60','A69','A70','A80','A99']

plt.bar(classi,lungh_media_seq_classe)
plt.title("Lunghezza media sequenze classi")
plt.xlabel("classi")
plt.ylabel("lunghezza media")
plt.show()

## diagramma a torta per vedere il numero di occorrenze nelle camera 2 e 3 in viste frontali

labels = ['cam 2', 'cam 3']
slices = [cont_2_2, cont_3_1]
plt.pie(slices, labels = labels, autopct = lambda x: f'{x:.1f}%')
plt.title("numero occorrenze camere 2 e 3 in vista frontale")
plt.show()

## bar per analizzare la lunghezza media delle classi nelle camere 2 e 3 in vf

fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax[0].bar(classi,lunghezza_seq_media_vf2)
ax[0].set_ylabel("lunghezza media VF2")
ax[0].set_title("Lunghezza Media Sequenze Classi in VF 2 e 3")

ax[1].bar(classi,lunghezza_seq_media_vf3)
ax[1].set_ylabel("lunghezza media VF3")
ax[1].set_xlabel("classi")

plt.show()

## analisi cross-view

# determino la percentuale di setting

setting = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']

plt.bar(setting,cont_lista_setting)
plt.ylabel("freq setting")
plt.xlabel("setting")
plt.title('Proporzione setting')
plt.show()

# bar per visualizzare la lunghezza media setting

plt.bar(setting,lungh_media_set)
plt.title("Lunghezza media setting")
plt.xlabel("setting")
plt.ylabel("lunghezza media")
plt.show()

# determino il numero di setting in vf 2 e 3

figure, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax[0].bar(setting,numero_setting_cam2)
ax[0].set_ylabel("frequenze setting vf 2")
ax[0].set_title("frequenza setting in vf 2 e 3")

ax[1].bar(setting,numero_setting_cam3)
ax[1].set_ylabel("frequenze setting vf 3")
ax[1].set_xlabel("setting")

plt.show()

# determino la lunghezza media dei setting in vf 2 e 3


figure, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)

ax[0].bar(setting,lungh_med_set_vf2)
ax[0].set_ylabel("lung media set vf 2")
ax[0].set_title("lunghezza media setting in vf 2 e 3")

ax[1].bar(setting,lungh_med_set_vf3)
ax[1].set_ylabel("lung media set vf 3")
ax[1].set_xlabel("setting")

plt.show()


plt.bar(setting, freq_set_vf2)
plt.ylabel("freq setting vf2")
plt.xlabel("setting")
plt.title("frequenza setting vf2")
plt.show()

# mostro l'animazione del movimento

for data in data_list:
    if 'skel_body0' in data:
        skel_body0 = data['skel_body0']
        skel_body_reshape = skel_body0.reshape(skel_body0.shape[0], -1)
        data['skel_body0'] = skel_body_reshape



for data in data_list:
    if "P001" in data['file_name']:
        if "A027" in data['file_name']:
            if "C002" in data['file_name']:
                if "R002" in data['file_name']:
                    if "S001" in data['file_name']:
                        positions = data['skel_body0']


connections = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (20, 8), (6, 7), (8, 9), (9, 10),
               (10, 11), (11, 23), (11, 24), (7, 22), (0, 12), (12, 13), (13, 14), (14, 15), (16, 17),
               (17, 18), (18, 19), (7, 21), (0, 16)]
dt = 1 / 30  # Freq Camp


def show_skeleton(positions, connections, dt, title=None, slowing_parameter=1):
    output_filename = "skeleton_animation_p2_set3_c3_r2.gif"

    if os.path.exists(output_filename):
        os.remove(output_filename)
    # Estraggo le coordinate X, Y, Z delle articolazioni
    x = positions[:, 0::3]
    y = positions[:, 1::3]
    z = positions[:, 2::3]

    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    ax.set_box_aspect([1, 1, 1])

    ax.view_init(elev = 90, azim = -90)

    def update(frame):
        ax.cla()  # Pulisce l'asse a ogni frame
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        if title:
            ax.set_title(title)
        ax.axis("off")

        # Plot skeleton connections
        for connection in connections:
            joint1_pos = (x[frame, connection[0]], y[frame, connection[0]], z[frame, connection[0]])
            joint2_pos = (x[frame, connection[1]], y[frame, connection[1]], z[frame, connection[1]])
            ax.plot([joint1_pos[0], joint2_pos[0]],
                    [joint1_pos[1], joint2_pos[1]],
                    [joint1_pos[2], joint2_pos[2]],
                    color="gray")

        # Plot joints
        ax.scatter(x[frame], y[frame], z[frame], color="blue", marker="o", s=10, alpha=0.7)

    # Uso FuncAnimation per generare l'animazione
    ani = FuncAnimation(fig, update, frames=positions.shape[0], interval=dt * 1000 * slowing_parameter)

    # Salvo l'animazione come file gif
    ani.save(output_filename, writer="imagemagick")
    plt.close()

#show_skeleton(positions, connections, dt=dt)


# salvo i dati in un dizionario
#dizionario = {}

#for data in data_list:
#    nome = data['file_name']
#    valore = data['skel_body0']
#    dizionario[nome] = valore

#with open('dizionario_mc.pkl', 'wb') as file:
    #pickle.dump(dizionario, file)
