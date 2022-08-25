from pathlib import Path
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

files = [
    'wsc_sbert.raw.search.rawpn.220823.1/result.pn..csv',
    'wsc_sbert.raw.sp.search.rawpn.sentp.220823.1/result.pn..csv',
    'wsc_sbert.bert-base-japanese-whole-word-masking.pn.220823.1/result.pn.cl-tohoku.bert-base-japanese-whole-word-masking.csv',
    'wsc_sbert.rinna-japanese-roberta-base.pn.220823.1/result.pn.rinna.japanese-roberta-base.csv',
    'wsc_sbert.nlp-waseda-roberta-base-japanese.pn.220823.1/result.pn.nlp-waseda.roberta-base-japanese.csv',
    'wsc_sbert.nlp-waseda-roberta-large-japanese.pn.220823.1/result.pn.nlp-waseda.roberta-large-japanese.csv',
    'xlm-roberta-base.pn.220823.1/result.pn.xlm-roberta-base.csv',
    'xlm-roberta-large.pn.220823.1/result.pn.xlm-roberta-large.csv',
    'wsc_sbert.t5-base-japanese-web.pn.220823.1.encoder/result.pn.megagonlabs.t5-base-japanese-web.csv',
    'wsc_sbert.t5-base-japanese-web.pn.220823.1.decoder/result.pn.megagonlabs.t5-base-japanese-web.csv',
    'wsc_sbert.rinna-japanese-gpt-1b.pn.220823.1/result.pn.rinna.japanese-gpt-1b.csv',
    'wsc_sbert.rinna-japanese-gpt2-medium.pn.220823.1/result.pn.rinna.japanese-gpt2-medium.csv',
    'wsc_sbert.bert-base-japanese-whole-word-masking.doublet.220823.1/result.doublet.cl-tohoku.bert-base-japanese-whole-word-masking.csv',
    'wsc_sbert.rinna-japanese-roberta-base.doublet.220823.1/result.doublet.rinna.japanese-roberta-base.csv',
    'wsc_sbert.nlp-waseda-roberta-base-japanese.doublet.220823.1/result.doublet.nlp-waseda.roberta-base-japanese.csv',
    'wsc_sbert.nlp-waseda-roberta-large-japanese.doublet.220823.1/result.doublet.nlp-waseda.roberta-large-japanese.csv',
    'xlm-roberta-base.doublet.220823.1/result.doublet.xlm-roberta-base.csv',
    'xlm-roberta-large.doublet.220823.1/result.doublet.xlm-roberta-large.csv',
    'wsc_sbert.t5-base-japanese-web.doublet.220823.1.encoder/result.doublet.megagonlabs.t5-base-japanese-web.csv',
    'wsc_sbert.t5-base-japanese-web.doublet.220823.1.decoder/result.doublet.megagonlabs.t5-base-japanese-web.csv',
    'wsc_sbert.rinna-japanese-gpt-1b.doublet.220823.1/result.doublet.rinna.japanese-gpt-1b.csv',
    'wsc_sbert.rinna-japanese-gpt2-medium.doublet.220823.1/result.doublet.rinna.japanese-gpt2-medium.csv',
    'xlm-roberta-large.doublet.yandx.220823.1/result.doublet.xlm-roberta-large.csv',
    'xlm-roberta-large.doublet.kawahara.220823.1/result.doublet.xlm-roberta-large.csv'
]

for file in files:
    df = pd.read_table(f'../../results/{file}', header=0, delimiter=',')
    dfs = df.sort_values(by=['dev_acc', 'test_acc'], ascending=False)
    print(f'{Decimal(str(dfs.iloc[0].test_acc * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}, {Decimal(str(dfs.iloc[0].dev_acc * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}, {int(dfs.iloc[0].epoch)}: {file}')



