import os
import shutil


SOURCE = 'jpg'
NUM_IMAGES_PER_FLOWER = 80
NUM_TRAIN_IMAGES = 65

FLOWER_NAMES = ['daffodil', 'snowdrop', 'lily_valley', 'bluebell', 'crocus', 'iris', \
    'tigerlily', 'tulip', 'fritillary', 'sunflower', 'daisy', 'colts_foot', 'dandelion', \
    'cowslip', 'buttercup', 'windflower', 'pansy']

os.mkdir('flowers')
os.mkdir(os.path.join('flowers', 'train'))
os.mkdir(os.path.join('flowers', 'val'))

for i in range(len(FLOWER_NAMES)):
    os.mkdir(os.path.join('flowers', 'train', FLOWER_NAMES[i]))
    os.mkdir(os.path.join('flowers', 'val', FLOWER_NAMES[i]))
    for j in range(NUM_IMAGES_PER_FLOWER):
        index = (i * 80) + j + 1
        index = str(index)
        index = ('0' * (4 - len(index))) + index
        filename = 'image_{}.jpg'.format(index)
        src = os.path.join(SOURCE, filename)

        if j < NUM_TRAIN_IMAGES:
            dest = os.path.join('flowers', 'train', FLOWER_NAMES[i], filename)
        else:
            dest = os.path.join('flowers', 'val', FLOWER_NAMES[i], filename)

        shutil.copyfile(src, dest)