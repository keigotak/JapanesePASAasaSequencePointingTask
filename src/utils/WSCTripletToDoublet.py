from pathlib import Path


modes = ['train', 'dev', 'test']
for mode in modes:
    with Path(f'../../data/Winograd-Schema-Challenge-Ja-master/{mode}-triplet.txt').open('r') as f:
        texts = f.raadlines()
    doublets = []
    for text in texts:
        items = text.strip().split('\t')
        doublets.append([items[1], items[2]])

