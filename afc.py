import random
import numpy as np


def afc(all_lesions, normals, contrasts, folds=1):
    print('Calculating 4afc...')
    # Shuffle
    # select 3 backgrouds, 1 lesion
    # calculate the result
    # repeat 1000 times
    # what should be passed?
    # lesion and background softmax values

    # Create database of the backgrounds ID and MD
    database = {}
    #               database
    #                  |
    # --ID#1----------ID#2-----------ID#3-------
    #     |            |               |
    #                  |
    # ---MD#1---------MD#2-----------MD#3-------
    #     |            |               |
    #                  |
    #-----------[arrayOfNormals]----------------

    #Loop through the backgrounds and sort into the database
    for f in normals:
        parse = f.split('_')
        ID = parse[2]
        MD = parse[11]
        if ID not in database:
            database[ID] = {}
        if MD not in database[ID]:
            database[ID][MD] = [normals[f]['soft']]
        else:
            database[ID][MD].append(normals[f]['soft'])
    # Print database stats
    #print('Printing database...\n')
    #for ID in database:
    #    print('ID - {}'.format(ID))
    #    for MD in database[ID]:
    #        print('  MD_{} - {}'.format(MD, len(database[ID][MD])))





    #lesions = {contrasts[0]:{name:{class, soft},
    #             contrasts[1]:{name:{class, soft},
    #             contrasts[2]:{name:{class, soft}}
    # normals = {name}
    all_normals = np.array([normals[f]['soft'] for f in normals])
    all_normals = np.squeeze(all_normals)
    num_correct = {contrasts[0]: 0,
                   contrasts[1]: 0,
                   contrasts[2]: 0}
    num_incorrect = {contrasts[0]: 0,
                   contrasts[1]: 0,
                   contrasts[2]: 0}
    # Get local density and granularity of the lesion
    # File name example:
    #
    #P_6CMBCM_L_7mm1005_SF_0.662_XYZ_442_203_1569_C_0.93_MD_22_RD_28_SDR_1_SDF
    #       _ -phantom ID                                __ - +-5
    #_1_LO_1_XYZ_505_184_400
    #
    # Background example name:
    # P_Tomo_6CMBCM_XYZ_985_204_292_XYD_933_221_MD_23_RD_28_SDR_1_SD_2_LD_2

    for contrast in all_lesions:
        for f in all_lesions[contrast]:
            lesion = all_lesions[contrast][f]['soft'][1]
            parse = f.split('_')
            ID = parse[1]
            MD = parse[13]
            #potential_normals = database[ID][MD]
            potential_normals = []
            try:
                potential_normals.extend(
                    database[ID][MD])
            except:
                pass
            MD_range = 1
            while len(potential_normals) < 3:
                try:
                    potential_normals.extend(
                        database[ID][str(int(MD)+MD_range)])
                except:
                    pass
                try:
                    potential_normals.extend(
                        database[ID][str(int(MD)-MD_range)])
                except:
                    pass
                MD_range += 1
            # Select 3 normals randomly
            random_indexes = random.sample(range(len(potential_normals)), 3)
            normal = [potential_normals[random_indexes[0]][1],
                      potential_normals[random_indexes[1]][1],
                      potential_normals[random_indexes[2]][1]]
            if max(lesion, max(normal)) == lesion:
                num_correct[contrast] += 1
            else:
                num_incorrect[contrast] += 1

    for contrast in num_correct:
        print('{}: {:.2f}%'.format(
            contrast,
            num_correct[contrast]
            / (num_correct[contrast]+num_incorrect[contrast])
            * 100))

def old_afc(all_lesions, normals, folds=1):
    print('Calculating 4afc...')
    # Shuffle
    # select 3 backgrouds, 1 lesion
    # calculate the result
    # repeat 1000 times
    # what should be passed?
    # lesion and background softmax values

    #lesions = {contrasts[0]:{name:{class, soft},
    #             contrasts[1]:{name:{class, soft},
    #             contrasts[2]:{name:{class, soft}}
    # normals = {name}
    all_normals = np.array([normals[f]['soft'] for f in normals])
    all_normals = np.squeeze(all_normals)
    print('all_normals.shape: {}'.format(all_normals.shape))
    num_correct = {contrasts[0]: 0,
                   contrasts[1]: 0,
                   contrasts[2]: 0}
    num_incorrect = {contrasts[0]: 0,
                   contrasts[1]: 0,
                   contrasts[2]: 0}
    print('all_normals: {}'.format(all_normals))
    #
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
