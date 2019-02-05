import numpy as np
import glob
from sklearn.model_selection import KFold

def load_dataset(dataset_path, n_folds, rand_state):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """

    # load datapath from path
    pos_path = glob.glob(dataset_path+'/1/img*')
    neg_path = glob.glob(dataset_path+'/0/img*')

    pos_num = len(pos_path)
    neg_num = len(neg_path)

    all_path = pos_path + neg_path

    #num_bag = len(all_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets