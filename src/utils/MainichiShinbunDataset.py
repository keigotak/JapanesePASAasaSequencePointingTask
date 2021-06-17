from path import Path
import random
random.seed(0)

filenames = ['mai1995.txt.utf8', 'mai94.txt.utf8', 'MAI98A.TXT.utf8', 'MAI98B.TXT.utf8', 'MAI99_1.TXT.utf8', 'MAI99_2.TXT.utf8']

for filename in filenames:
    with (Path('/cldata/mainichi') / filename).open('r') as f:
        texts = f.readlines()

    datasets = []
    counts, flg_new_sentence = 0, False
    for text in texts:
        if text.startswith('＼Ｔ１＼'):
            datasets.append(text.strip('＼Ｔ１＼'))
        elif text.startswith('＼Ｔ２＼'):
            datasets.append(text.strip('＼Ｔ２＼'))

random.shuffle(datasets)

mainichi_path = Path('../../data/mainichi/contents.txt.shuffle')
with mainichi_path.open('w') as f:
    f.writelines(datasets)
