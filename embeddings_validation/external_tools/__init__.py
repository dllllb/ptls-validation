"""Tools for external tasks
"""
import json
import os

from omegaconf import OmegaConf

from embeddings_validation.config import Config
from embeddings_validation.file_reader import TargetFile
from embeddings_validation.tasks.fold_splitter import FoldSplitter


def get_fold_ids(
    config_path,
    kind,
    fold_id,
):
    """Read id's from file and returns them as a list
    """
    conf= OmegaConf.load(config_path)
    conf = Config.get_conf(conf)

    fold_split_index = FoldSplitter(conf=conf).output().path

    with open(fold_split_index, 'r') as f:
        folds = json.load(f)
    current_fold = folds[str(fold_id)]
    current_part = current_fold[kind]
    id_file_path = current_part["path"]
    df_targets = TargetFile.load(id_file_path)
    # print(f'====Loaded {kind} for fold_id={fold_id}')
    return df_targets.ids_values


def get_fold_list(
    config_path,
):
    """Returns the list of fold ids
    """
    conf= OmegaConf.load(config_path)
    conf = Config.get_conf(conf)
    return conf.folds


# if __name__ == '__main__':
#     ids = get_fold_list(
#         'conf/embeddings_validation_baselines_supervised.yaml',
#     )
#     print(ids)
