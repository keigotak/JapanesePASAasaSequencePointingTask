import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ptr = {'5000': [
    0.7536841434,
    0.7515462561,
    0.749273623,
    0.7631763208,
    0.7601571468
],
    '10000': [
        0.7807561449,
        0.7811027109,
        0.7779775281,
        0.7778828735,
        0.762617546
    ],
    '20000': [
        0.8154835005,
        0.8099400009,
        0.8128145291,
        0.8126550469,
        0.8130392793
    ],
    '40000': [
        0.8324018838,
        0.832880263,
        0.837285986,
        0.8333386262,
        0.8343924879
    ],
    '62489': [
        0.8436204948,
        0.8458810736,
        0.8477854221,
        0.8439502033,
        0.8419502462
    ]}

seq = {'5000': [
    0.7452312258,
    0.7397103159,
    0.7466504663,
    0.7457909285,
    0.7518466437
],
    '10000': [
        0.7568174854,
        0.7444252147,
        0.7668869982,
        0.7741339042,
        0.7226954775
    ],
    '20000': [
        0.7991629063,
        0.803735091,
        0.7999292695,
        0.8101807802,
        0.8051910645
    ],
    '40000': [
        0.8312304574,
        0.8278093205,
        0.8291391573,
        0.8333359907,
        0.8293593199
    ],
    '62489': [
        0.8427781346,
        0.8387178233,
        0.8427916869,
        0.8424965948,
        0.8439870498
    ]}

items = []
a_values = []
b_values = []
c_values = []
for item in ['5000', '10000', '20000', '40000', '62489']:
    a_values.extend(seq[item])
    b_values.extend([int(item)] * len(seq[item]))
    c_values.extend(['Baseline'] * len(seq[item]))
    a_values.extend(ptr[item])
    b_values.extend([int(item)] * len(ptr[item]))
    c_values.extend(['Proposed (global-argmax)'] * len(ptr[item]))
my_dict = {'F1': a_values,
           'Data size': b_values,
           'Type': c_values}

df = pd.DataFrame.from_dict(my_dict)

plt.ylim(0.72, 0.85)
sns.set_style("whitegrid")
ax = sns.barplot(x="Data size", y="F1", hue="Type", data=my_dict, capsize=.2, errwidth=0.5, palette="Blues", linewidth=0.3, edgecolor="navy")
ax.set(xlabel='# of training data', ylabel='F1 score')
ax.set_xlabel("# of training data", fontsize=12)
ax.set_ylabel("F1 score", fontsize=12)
ax.tick_params(labelsize=11)
plt.savefig(str(Path('../../tex/pointernet/pic30.png').resolve()), bbox_inches="tight")


