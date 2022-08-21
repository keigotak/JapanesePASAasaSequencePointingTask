import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
# from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import dask.array as da
from dask_ml.decomposition import PCA
from SimilarlityBERTDoubletEvaluate import get_properties

dirs = ['wsc_sbert.rinna-japanese-gpt2-medium.doublet.220317',
        'wsc_sbert.t5-base-japanese-web.doublet.220317',
        'wsc_sbert.bert-base-japanese-whole-word-masking.doublet.220317',
        'wsc_sbert.nlp-waseda-roberta-base-japanese.doublet.220317',
        'wsc_sbert.rinna-japanese-gpt-1b.doublet.220317',
        'wsc_sbert.rinna-japanese-roberta-base.doublet.220317']

i = 5
run_modes = ['rinna-gpt2', 'tohoku-bert', 't5-base', 'rinna-roberta', 'nlp-waseda-roberta-base-japanese', 'rinna-japanese-gpt-1b']
model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_modes[i])

positive_embeddings, negative_embeddings, scores = [], [], []
with Path(f'{OUTPUT_PATH}.220317/details.results.doublet.{model_name.replace("/", ".")}.raw.pt').open('rb') as f:
    rets = torch.load(f)
# anchor_embeddings.append(rets['anchor_embeddings'])
for ret in rets['results']:
    positive_embeddings.append(ret[6] if ret[3] == 1 else ret[5])
    negative_embeddings.append(ret[5] if ret[3] == 1 else ret[6])
    # scores.append(rets['accuracy_euclidean'])

pca = PCA(n_components=2)

X = np.concatenate([positive_embeddings, negative_embeddings])
dX = da.from_array(X, chunks=X.shape)
pca = PCA(n_components=2)
pca.fit(dX)
# pca_anchor_embedding = pca.transform(anchor_embeddings[i]).compute()
pca_positive_embedding = pca.transform(positive_embeddings).compute()
pca_negative_embedding = pca.transform(negative_embeddings).compute()

text_model = 'BERT'
if 'gpt' in model_name:
    text_model = 'GPT-2'
elif 't5' in model_name:
    text_model = 'T5'
elif 'roberta' in model_name:
    text_model = 'RoBEERTa'

mode = 'scatter'
if mode == 'scatter':
    pca_X = np.concatenate([pca_positive_embedding, pca_negative_embedding]).tolist()
    pca_Y = ['positive'] * pca_positive_embedding.shape[0] + ['negative'] * pca_negative_embedding.shape[0]
    data = [x + [y] for x, y in zip(pca_X, pca_Y)]
    data = pd.DataFrame(data=data, columns=['pc1', 'pc2', 'category'])
    g = sns.scatterplot(data=data, x='pc1', y='pc2', hue='category', palette='bright')

    # weights_anchor = pca_anchor_embedding.mean(axis=0).tolist()
    weights_positive = pca_positive_embedding.mean(axis=0).tolist()
    weights_negative = pca_negative_embedding.mean(axis=0).tolist()

    # plt.plot(weights_anchor[0], weights_anchor[1], marker='^', markersize=9, c='k')
    plt.plot(weights_positive[0], weights_positive[1], marker='s', markersize=9, c='k')
    plt.plot(weights_negative[0], weights_negative[1], marker='p', markersize=9, c='k')

    plt.savefig(f'./scatterplot.doublet.{model_name.replace("/", ".")}.raw.png')
else:
    weights_anchor = pca_anchor_embedding.mean(axis=0).tolist()
    weights_positive = pca_positive_embedding.mean(axis=0).tolist()
    weights_negative = pca_negative_embedding.mean(axis=0).tolist()

    print(f'anchor: {weights_anchor[0]}, {weights_anchor[1]}')
    print(f'positive: {weights_positive[0]}, {weights_positive[1]}')
    print(f'negative: {weights_negative[0]}, {weights_negative[1]}')

'''
'''