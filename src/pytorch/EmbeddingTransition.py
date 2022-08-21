import torch
from pathlib import Path
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
import pandas as pd

dirs = ['wsc_sbert.rinna-japanese-gpt2-medium.triplet.test.220212/eval',
        'wsc_sbert.t5-base-japanese-web.triplet.test.220213',
        'wsc_sbert.bert-base-japanese-whole-word-masking.triplet.test.220212/eval']

epochs = 100
model = dirs[2]
positive_distances, negative_distances, scores = [], [], []
for i in range(epochs):
    with Path(f'../../results/{model}/epoch{i}.pkl').open('rb') as f:
        rets = torch.load(f)
    positive_distances.append(paired_euclidean_distances(rets['anchor_embeddings'], rets['positive_embeddings']))
    negative_distances.append(paired_euclidean_distances(rets['anchor_embeddings'], rets['negative_embeddings']))
    scores.append(rets['accuracy_euclidean'])

mean_positive_distances, mean_negative_distances = [], []
with Path(f'./embeddings.{model.split("/")[0]}.tsv').open('w') as f:
    f.write(f'epoch\tmean_positive_distance\tmean_negative_distance\tscore\n')
    for i in range(epochs):
        mean_positive_distance = sum(positive_distances[i]) / len(positive_distances[i])
        mean_negative_distance = sum(negative_distances[i]) / len(negative_distances[i])
        f.write(f'{i}\t{mean_positive_distance}\t{mean_negative_distance}\t{scores[i]}\n')
        mean_positive_distances.append(mean_positive_distance)
        mean_negative_distances.append(mean_negative_distance)

text_model = 'BERT'
if 'gpt2' in model:
    text_model = 'GPT-2'
elif 't5' in model:
    text_model = 'T5'
# data = [[e+1, p, s, 'positive'] for e, p, s in zip(range(epochs), mean_positive_distances, scores)] + [[e+1, p, s, 'negative'] for e, p, s in zip(range(epochs), mean_negative_distances, scores)]
# data = pd.DataFrame(data=data, columns=['epoch', 'distance', 'accuracy', 'category'])
data = [[e+1, p, n, p-n, s] for e, p, n, s in zip(range(epochs), mean_positive_distances, mean_negative_distances, scores)]
data = pd.DataFrame(data=data, columns=['epoch', 'positive distance', 'negative distance', 'diff', 'accuracy'])


# g = sns.relplot(data=data, x="epoch", y='distance', hue='category',
#                 kind="line", palette='bright', markers=True, dashes=False)
# data = [[e+1, s] for e, s in zip(range(epochs), scores)]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(data['epoch'].tolist(), data['positive distance'].tolist(), label='anchor-positive')
ax1.plot(data['epoch'].tolist(), data['negative distance'].tolist(), label='anchor-negative')
ax1.set_xlabel('epoch')
ax1.set_ylabel('euclidean distance from anchor')
h1, l1 = ax1.get_legend_handles_labels()

ax2.set_ylim(0.0, 1.0)
ax2.plot(data['epoch'].tolist(), data['accuracy'].tolist(), label='accuracy', c='red')
ax2.set_ylabel('accuracy')
h2, l2 = ax2.get_legend_handles_labels()

ax1.legend(h1 + h2, l1 + l2)
# plt.title(f'model: {text_model}')
plt.savefig(f'./relplot.{model.split("/")[0]}.png')
