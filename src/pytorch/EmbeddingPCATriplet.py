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

dirs = ['wsc_sbert.rinna-japanese-gpt2-medium.triplet.test.220212/eval',
        'wsc_sbert.t5-base-japanese-web.triplet.test.220212',
        'wsc_sbert.bert-base-japanese-whole-word-masking.triplet.test.220212/eval']

epochs = [0, 99]
model = dirs[0]
anchor_embeddings, positive_embeddings, negative_embeddings, scores = [], [], [], []
for i in epochs:
    with Path(f'../../results/{model}/epoch{i}.pkl').open('rb') as f:
        rets = torch.load(f)
    anchor_embeddings.append(rets['anchor_embeddings'])
    positive_embeddings.append(rets['positive_embeddings'])
    negative_embeddings.append(rets['negative_embeddings'])
    scores.append(rets['accuracy_euclidean'])

pca = PCA(n_components=2)

i = 1
X = np.concatenate([anchor_embeddings[i], positive_embeddings[i], negative_embeddings[i]])
dX = da.from_array(X, chunks=X.shape)
pca = PCA(n_components=2)
pca.fit(dX)
pca_anchor_embedding = pca.transform(anchor_embeddings[i]).compute()
pca_positive_embedding = pca.transform(positive_embeddings[i]).compute()
pca_negative_embedding = pca.transform(negative_embeddings[i]).compute()

text_model = 'BERT'
if 'gpt2' in model:
    text_model = 'GPT-2'
elif 't5' in model:
    text_model = 'T5'
text_epoch = 1
if i == 1:
    text_epoch = 100

mode = 'scatter'
if mode == 'scatter':
    pca_X = np.concatenate([pca_anchor_embedding, pca_positive_embedding, pca_negative_embedding]).tolist()
    pca_Y = ['anchor'] * pca_anchor_embedding.shape[0] + ['positive'] * pca_positive_embedding.shape[0] + ['negative'] * pca_negative_embedding.shape[0]
    data = [x + [y] for x, y in zip(pca_X, pca_Y)]
    data = pd.DataFrame(data=data, columns=['pc1', 'pc2', 'category'])
    g = sns.scatterplot(data=data, x='pc1', y='pc2', hue='category', palette='bright')

    weights_anchor = pca_anchor_embedding.mean(axis=0).tolist()
    weights_positive = pca_positive_embedding.mean(axis=0).tolist()
    weights_negative = pca_negative_embedding.mean(axis=0).tolist()

    plt.plot(weights_anchor[0], weights_anchor[1], marker='^', markersize=9, c='k')
    plt.plot(weights_positive[0], weights_positive[1], marker='s', markersize=9, c='k')
    plt.plot(weights_negative[0], weights_negative[1], marker='p', markersize=9, c='k')

    # plt.annotate("anc. mean",
    #              xy = (weights_anchor[0], weights_anchor[1]), xycoords = 'data',
    #              xytext = (10, -7.7), textcoords = 'data',
    #              arrowprops = dict(arrowstyle="->", connectionstyle="arc3", color='black'),
    #             )
    # plt.annotate("pos. mean",
    #              xy = (weights_positive[0], weights_positive[1]), xycoords = 'data',
    #              xytext = (3, -7.7), textcoords = 'data',
    #              arrowprops = dict(arrowstyle="->", connectionstyle="arc3", color='black'),
    #             )
    # plt.annotate("neg. mean",
    #              xy = (weights_negative[0], weights_negative[1]), xycoords = 'data',
    #              xytext = (9, -7.5), textcoords = 'data',
    #              arrowprops = dict(arrowstyle="->", connectionstyle="arc3", color='black'),
    #             )

    # plt.savefig(f'./scatterplot.{model.split("/")[0]}-{i}.eps', format='eps')
    plt.savefig(f'./scatterplot.{model.split("/")[0]}-{i}.png')
else:
    weights_anchor = pca_anchor_embedding.mean(axis=0).tolist()
    weights_positive = pca_positive_embedding.mean(axis=0).tolist()
    weights_negative = pca_negative_embedding.mean(axis=0).tolist()

    print(f'anchor: {weights_anchor[0]}, {weights_anchor[1]}')
    print(f'positive: {weights_positive[0]}, {weights_positive[1]}')
    print(f'negative: {weights_negative[0]}, {weights_negative[1]}')

'''
GPT-2
epoch: 1
anchor: 0.0022954114247113466, 0.17967736721038818
positive: 0.001467792084440589, -0.08828655630350113
negative: -0.0037687295116484165, -0.09139396250247955

epoch: 100
anchor: 4.524056434631348, -0.004430854227393866
positive: -1.2753205299377441, -0.0153373247012496
negative: -3.248737096786499, 0.01976458542048931

T5
epoch: 1
anchor: -0.23966985940933228, -0.09020841121673584
positive: 0.06975245475769043, 0.04531823843717575
negative: 0.1698967069387436, 0.04488862305879593

epoch: 100
anchor: -0.22786544263362885, -0.10111027956008911
positive: 0.1103484109044075, 0.05383714661002159
negative: 0.11749574542045593, 0.04724394530057907

BERT
epoch: 1
anchor: -0.013629992492496967, 0.007517729885876179
positive: 0.008163467049598694, -0.004562306217849255
negative: 0.005466565024107695, -0.0029558439273387194

epoch: 100
anchor: -0.1264837235212326, 2.5248219966888428
positive: 0.021674789488315582, -0.5938087701797485
negative: 0.10480891913175583, -1.9310132265090942
'''