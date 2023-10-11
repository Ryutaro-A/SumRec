import itertools
from operator import itemgetter
import torch
from transformers import AutoTokenizer
import numpy as np




def get_token(target):
    # input_ids = target[0]["input_ids"]
    # token_type_ids = target[0]["token_type_ids"]
    # attention_mask = target[0]["attention_mask"]
    input_ids = [data["input_ids"] for data in target]
    token_type_ids = [data["token_type_ids"] for data in target]
    attention_mask = [data["attention_mask"] for data in target]
    return {
        "input_ids": torch.stack(input_ids),
        "token_type_ids": torch.stack(token_type_ids),
        "attention_mask": torch.stack(attention_mask)
    }

def collate_fn(batch):
    dialogue, desc, score, label, graph, info = list(zip(*batch))

    return {
        "dialogue": get_token(dialogue),
        "desc": get_token(desc),
        "score": score,
        "label": torch.stack(label) if label[0] is not None else None,
        "graph": torch.tensor(graph) if graph[-1] is not None else None,
        "info": info
    }

def collate_fn_one_encoder(batch):
    dialogue_desc, _, score, label, _, info = list(zip(*batch))

    input_dict = get_token(dialogue_desc)
    # input_dict = {key: dialogue_dict[key] for key in input_dict.keys()}

    # print(self.tokenizer.decode(input_dict["input_ids"][0]))
    # exit()

    collated_data = {
        "dialogue": input_dict,
        "score": score,
        "info":info
    }


    return collated_data