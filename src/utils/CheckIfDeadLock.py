import os
from time import sleep
from pathlib import Path


def get_access(path):
    accessed = False
    if path.exists():
        for i in range(5):
            try:
                os.rename(str(path.resolve()), str(path.resolve()))
            except OSError as e:
                sleep(1)
            else:
                accessed = True
                break
    return accessed


if __name__ == "__main__":
    tags = ['lstm190201', 'pointer190201']
    for item in tags:
        if get_access(Path('../../results/trials').joinpath(item + '.pkl')):
            print('{} is not deadlocked.'.format(item))
    print('check end.')