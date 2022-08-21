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
        'wsc_sbert.rinna-japanese-gpt-1b.doublet.220317',
        'wsc_sbert.bert-base-japanese-whole-word-masking.doublet.220317',
        'wsc_sbert.nlp-waseda-roberta-base-japanese.doublet.220317',
        'wsc_sbert.rinna-japanese-roberta-base.doublet.220317',
        'wsc_sbert.t5-base-japanese-web.doublet.220317']

run_modes = ['rinna-gpt2', 'rinna-japanese-gpt-1b', 'tohoku-bert', 'nlp-waseda-roberta-base-japanese', 'rinna-roberta', 't5-base']
texts = [['id', 'positive_sentence', 'negative_sentence'] + run_modes]
for i, run_mode in enumerate(run_modes):
    model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)

    with Path(f'{OUTPUT_PATH}.220317/details.results.doublet.{model_name.replace("/", ".")}.pt').open('rb') as f:
        rets = torch.load(f)
    # anchor_embeddings.append(rets['anchor_embeddings'])
    if i == 0:
        for ret in rets['results']:
            if ret[3] == 0:
                texts.append([ret[0], ret[1], ret[2], ret[4]])
            else:
                texts.append([ret[0], ret[2], ret[1], int(not bool(ret[4]))])
    else:
        for j, ret in enumerate(rets['results']):
            if ret[3] == 0:
                texts[j+1].extend([ret[4]])
            else:
                texts[j+1].extend([int(not bool(ret[4]))])

with Path(f'summary.doublet.220317.csv').open('w') as f:
    for text in texts:
        f.write(','.join(map(str, text)) + '\n')
