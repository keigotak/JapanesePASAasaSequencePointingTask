from path import Path


def count_duplication(path, index=0):
    with path.open('r', encoding='utf-8') as f:
        texts = f.readlines()
    headers = texts[0]
    contents = texts[1:]

    sentence = contents[0].split(',')[14]
    predicate = contents[0].split(',')[2]

    duplicate_sentence = []
    labelwise_counter = {}
    rets = []
    dup_flg = False
    for content in contents:
        content = content.strip().split(',')
        if content[14] != sentence or content[2] != predicate:
            if dup_flg:
                labelwise_counter = {key: val for key, val in labelwise_counter.items() if val > 1}
                duplicate_sentence.extend([[sentence, rets, labelwise_counter]])
            dup_flg = False
            sentence = content[14]
            predicate = content[2]
            labelwise_counter = {}
            rets = []
        prediction = content[8].strip()
        if prediction != '3':
            if prediction not in labelwise_counter.keys():
                labelwise_counter[prediction] = 1
            else:
                labelwise_counter[prediction] += 1
                dup_flg = True
        rets.extend([content])
    print('{}: {}'.format(path.name, len(duplicate_sentence)))


def main():
    files = ['acmslntcglove', 'acmspgntcglove', 'acmsplntcglove', 'acmspnntcglove']
    for file in files:
        path = Path('../../results/detaillog_{}.txt'.format(file))
        count_duplication(path)


if __name__ == '__main__':
    main()
