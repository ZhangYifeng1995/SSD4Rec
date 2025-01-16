import copy
import importlib
import os
import pickle
import warnings
from typing import Literal
import pandas as pd

from recbole.data.dataloader import *
from recbole.data.utils import load_split_dataloaders, create_samplers, get_dataloader, save_split_dataloaders
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils.argument_list import dataset_arguments

import numpy as np
import torch

from recbole.data.dataset import SequentialDataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource
import torch.nn.utils.rnn as rnn_utils

def _convert_to_tensor(data):
    elem = data[0]
    if isinstance(elem, (float, int, np.float, np.int64)):
        new_data = torch.as_tensor(data)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        seq_data = [torch.as_tensor(d) for d in data]
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
    else:
        raise ValueError(f"[{type(elem)}] is not supported!")
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    return new_data

class Customized_Interaction(Interaction):
    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = value                    # change here
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(
                        f"The type of {key}[{type(value)}] is not supported!"
                    )
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(
                f"[{type(interaction)}] is not supported for initialize `Interaction`!"
            )
        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, len(self.interaction[k])) # change here

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        if isinstance(index, (np.ndarray, torch.Tensor)):
            index = index.tolist()

        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k][index]
        return Customized_Interaction(ret) # chage here
    
    def __str__(self):
        info = [f"The batch_size of interaction: {self.length}"]
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.dtype}"
            info.append(temp_str)
        info.append("\n")
        return "\n".join(info)

class SSD4RecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def _dataframe_to_interaction(self, data):
        new_data = {}
        for k in data:
            value = data[k].values
            ftype = self.field2type[k]
            if ftype == FeatureType.TOKEN:
                new_data[k] = torch.LongTensor(value)
            elif ftype == FeatureType.FLOAT:
                if k in self.config["numerical_features"]:
                    new_data[k] = torch.FloatTensor(value.tolist())
                else:
                    new_data[k] = torch.FloatTensor(value)
            elif ftype == FeatureType.TOKEN_SEQ:
                seq_data = [torch.LongTensor(d[: self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                if k in self.config["numerical_features"]:
                    base = [
                        torch.FloatTensor(d[0][: self.field2seqlen[k]]) for d in value
                    ]
                    base = rnn_utils.pad_sequence(base, batch_first=True)
                    index = [
                        torch.FloatTensor(d[1][: self.field2seqlen[k]]) for d in value
                    ]
                    index = rnn_utils.pad_sequence(index, batch_first=True)
                    new_data[k] = torch.stack([base, index], dim=-1)
                else:
                    seq_data = [
                        torch.FloatTensor(d[: self.field2seqlen[k]]) for d in value
                    ]
                    new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Customized_Interaction(new_data)

    def data_augmentation(self):
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"] # 200
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if self.config['var_len'] == False and (i - seq_start > max_item_list_len):  # Limit the length
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        # new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")

                new_list = []
                value = self.inter_feat[field]
                for i, index in enumerate(item_list_index):
                    new_list.append(value[index])

                new_list = np.array(new_list, dtype=object)
                new_dict[list_field] = new_list

        for k in new_dict:
            new_data[k] = new_dict[k]

        self.inter_feat = new_data


class SSD4RecTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.mask_ratio = config['maskratio']

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        
        item_id_list = data.interaction['item_id_list']
        item_id = data.interaction['item_id']
        item_length = data.interaction['item_length']

        item_id_list = torch.cat(list(item_id_list), dim=0)
        item_idx = torch.cat([torch.full((item_length[i], ), i, dtype=torch.int32) for i in range(len(item_length))], dim=0)
        cum_item_length = item_length.cumsum(dim=0) # rememeber to -1 latter

        # mask_index
        mask_index = (torch.rand(item_id_list.shape) > self.mask_ratio)

        # flip_index
        flip_index = []
        start = -1
        for end in cum_item_length - 1:
            flip_index += range(end,start,-1)
            start = end
        flip_index = torch.tensor(flip_index)

        item_id_list = item_id_list * mask_index

        return item_id, item_id_list, cum_item_length, item_idx, flip_index
    

class SSD4RecFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        inter_num = len(data)
        positive_u = torch.arange(inter_num)

        item_id_list = data.interaction['item_id_list']
        positive_i = data.interaction['item_id']
        item_length = data.interaction['item_length']

        item_id_list = torch.cat(list(item_id_list), dim=0)
        cum_item_length = item_length.cumsum(dim=0) # rememeber to -1 latter
        item_idx = torch.cat([torch.full((item_length[i], ), i, dtype=torch.int32) for i in range(len(item_length))], dim=0)

        # flip_index
        flip_index = []
        start = -1
        for end in cum_item_length - 1:
            flip_index += range(end,start,-1)
            start = end
        flip_index = torch.tensor(flip_index)
        
        return item_id_list, cum_item_length, item_idx, flip_index, positive_u, positive_i

def SSD4RecData_preparation(config, dataset):
    model_type = config["MODEL_TYPE"]
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets

    train_sampler, valid_sampler, test_sampler = create_samplers(
        config, dataset, built_datasets
    )
    train_data = SSD4RecTrainDataLoader( # chage here
        config, train_dataset, train_sampler, shuffle=config["shuffle"]
    )

    valid_data = SSD4RecFullSortEvalDataLoader(
        config, valid_dataset, valid_sampler, shuffle=False
    )
    test_data = SSD4RecFullSortEvalDataLoader(
        config, test_dataset, test_sampler, shuffle=False
    )
    if config["save_dataloaders"]:
        save_split_dataloaders(
            config, dataloaders=(train_data, valid_data, test_data)
        )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data