
import itertools
from operator import itemgetter
import torch
from transformers import AutoTokenizer
import numpy as np


def get_pred(src, model, device):
    input_ids = src["input_ids"].to(device)
    token_type_ids = src["token_type_ids"].to(device)
    attention_mask = src["attention_mask"].to(device)
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    logits = output["logits"].squeeze()
    pred = torch.tanh(logits)
    return pred
    #return logits


def stack_data(data_list):
    dialogue_input_ids_list = [data["dialogue"]["input_ids"] for data in data_list]
    dialogue_token_type_ids_list = [data["dialogue"]["token_type_ids"] for data in data_list]
    dialogue_attention_mask_list = [data["dialogue"]["attention_mask"] for data in data_list]

    desc_input_ids_list = [data["desc"]["input_ids"] for data in data_list]
    desc_token_type_ids_list = [data["desc"]["token_type_ids"] for data in data_list]
    desc_attention_mask_list = [data["desc"]["attention_mask"] for data in data_list]

    score_list = [data["score"] for data in data_list]
    spot_id_list = [data["info"]["id"] for data in data_list]

    dialogue_input_ids = torch.stack(dialogue_input_ids_list).squeeze()
    dialogue_token_type_ids = torch.stack(dialogue_token_type_ids_list).squeeze()
    dialogue_attention_mask = torch.stack(dialogue_attention_mask_list).squeeze()

    desc_input_ids = torch.stack(desc_input_ids_list).squeeze()
    desc_token_type_ids = torch.stack(desc_token_type_ids_list).squeeze()
    desc_attention_mask = torch.stack(desc_attention_mask_list).squeeze()

    dialogue = {
        "input_ids": dialogue_input_ids,
        "token_type_ids": dialogue_token_type_ids,
        "attention_mask": dialogue_attention_mask
    }
    desc = {
        "input_ids": desc_input_ids,
        "token_type_ids": desc_token_type_ids,
        "attention_mask": desc_attention_mask
    }

    score = torch.stack(score_list).squeeze()
    return dialogue, desc, score, spot_id_list

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result


