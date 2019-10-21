import matplotlib.pyplot as plt
from Utils import *

path = "D:\Rossi\data\data_part_107part_val/2008_000142.png"
seg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

classes = listPartsNames()
classes0 = classes[65:73]
classes1 = classes[77:89]
classes = classes0 + classes1
adj = adj_func_prova_2(seg)

adj0 = adj[65:73, 65:73]
adj1 = adj[77:89, 77:89]

adj = np.block([[adj0, np.zeros((adj0.shape[1], adj1.shape[1]))], [np.zeros((adj1.shape[1], adj0.shape[1])), adj1]])

plt.matshow(adj, cmap='Blues')
tick_marks = np.arange(len(classes))
plt.xticks(np.arange(len(classes)), classes, rotation=90, fontsize=9)
plt.yticks(tick_marks, classes, fontsize=9)
plt.gcf().subplots_adjust(bottom=0.15)

thresh = adj.max() / 2.
for i, j in itertools.product(range(adj.shape[0]), range(adj.shape[1])):
    plt.text(j, i, np.round(adj[i, j], 2), horizontalalignment="center",
             color="white" if adj[i, j] > thresh else "black", fontsize=7)

plt.show()
