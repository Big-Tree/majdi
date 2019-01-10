import random
import numpy as np


def afc_real(all_lesions, normals, folds=1):
    print('Calculating 4afc...')
    # Shuffle
    # select 3 backgrouds, 1 lesion
    # calculate the result
    # repeat for every lesion
    # what should be passed?
    # lesion and background softmax values

    #lesions = {'0.95':{name:{class, soft},
    #             '0.97':{name:{class, soft},
    #             '0.99':{name:{class, soft}}
    # normals = {name}
    all_normals = np.array([normals[f]['soft'] for f in normals])
    all_normals = np.squeeze(all_normals)
    print('all_normals.shape: {}'.format(all_normals.shape))
    num_correct = {'0.95': 0,
                   '0.97': 0,
                   '0.99': 0}
    num_incorrect = {'0.95': 0,
                   '0.97': 0,
                   '0.99': 0}
    print('all_normals: {}'.format(all_normals))

    for contrast in all_lesions:
        for f in all_lesions[contrast]:
            lesion = all_lesions[contrast][f]['soft'][1]
            # select 1 lesion and three normals
            #random.shuffle(all_normals)
            np.random.shuffle(all_normals)
            normal = [all_normals[0][1],
                      all_normals[1][1],
                      all_normals[2][1]]
            print('normal soft values: {}'.format(normal))
            print('lesion soft values: {}'.format(lesion))
            if max(lesion, max(normal)) == lesion:
                num_correct[contrast] += 1
            else:
                num_incorrect[contrast] += 1
    print('len(all_normals): {}'.format(len(all_normals)))
    print('num_correct:\n{}'.format(num_correct))
    print('num_incorrect:\n{}'.format(num_incorrect))

    for contrast in num_correct:
        print('{}: {:.2f}%'.format(
            contrast,
            num_correct[contrast]
            / (num_correct[contrast]+num_incorrect[contrast])
            * 100))
def afc(all_lesions, normals, folds=1):
    print('Calculating 4afc...')
    # Shuffle
    # select 3 backgrouds, 1 lesion
    # calculate the result
    # repeat 1000 times
    # what should be passed?
    # lesion and background softmax values

    #lesions = {'0.95':{name:{class, soft},
    #             '0.97':{name:{class, soft},
    #             '0.99':{name:{class, soft}}
    # normals = {name}
    all_normals = np.array([normals[f]['soft'] for f in normals])
    all_normals = np.squeeze(all_normals)
    print('all_normals.shape: {}'.format(all_normals.shape))
    num_correct = {'0.95': 0,
                   '0.97': 0,
                   '0.99': 0}
    num_incorrect = {'0.95': 0,
                   '0.97': 0,
                   '0.99': 0}
    print('all_normals: {}'.format(all_normals))

    for contrast in all_lesions:
        for f in all_lesions[contrast]:
            lesion = all_lesions[contrast][f]['soft'][1]
            # select 1 lesion and three normals
            #random.shuffle(all_normals)
            np.random.shuffle(all_normals)
            normal = [all_normals[0][1],
                      all_normals[1][1],
                      all_normals[2][1]]
            print('normal soft values: {}'.format(normal))
            print('lesion soft values: {}'.format(lesion))
            if max(lesion, max(normal)) == lesion:
                num_correct[contrast] += 1
            else:
                num_incorrect[contrast] += 1
    print('len(all_normals): {}'.format(len(all_normals)))
    print('num_correct:\n{}'.format(num_correct))
    print('num_incorrect:\n{}'.format(num_incorrect))

    for contrast in num_correct:
        print('{}: {:.2f}%'.format(
            contrast,
            num_correct[contrast]
            / (num_correct[contrast]+num_incorrect[contrast])
            * 100))
