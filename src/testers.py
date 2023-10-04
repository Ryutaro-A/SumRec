import os
import time
import argparse
from socket import gethostname
import sys
import json
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

import transformers
from transformers import TrainingArguments
from prepare_data import MyDataset, BertBiEncoderCollator, GraphBERTCollator, BertOneEncoderCollator
from model import BertBiEncoder, BertBiEncoderWithLabel, GraphLinearBERT, GraphBERT, DoubleBERTEncoder, DoubleBERTEncoderWithLabel, OneEncoder
from utils.input_data import collate_fn, collate_fn_one_encoder
from utils.others import stack_data
from transformers import Trainer
from transformers import EarlyStoppingCallback
from transformers import logging
from transformers import BertJapaneseTokenizer, AutoTokenizer

logging.set_verbosity_warning()

import numpy as np



def pre_process(
    args,
    tmp_output_dir: str,
    result_dir: str,
):
    test_dataset = MyDataset(args, exp_type="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_one_encoder if args.one_encoder else collate_fn)

    # tokenizer = BertJapaneseTokenizer.from_pretrained(args.plm_vocab_dir)
    # print("\n".join([tokenizer.decode(data[0]["input_ids"]).replace("[PAD]", "").replace(" ", "") for data in train_dataset]))
    # exit()

    # model
    # config = transformers.AutoConfig.from_pretrained(f"./configs/{args.pretrained_model_name.split('/')[-1]}.json")
    config = transformers.AutoConfig.from_pretrained(args.pretrained_model_name)

    criterion = nn.MSELoss(reduction="mean")


    if args.use_double_encoder:
        if args.use_graph_emb:
            print("未実装")
            exit()
        else:
            if args.use_label:
                model = DoubleBERTEncoderWithLabel(config, args, criterion)
            else:
                model = DoubleBERTEncoder(config, args, criterion)
    elif args.one_encoder:
        model = OneEncoder(config, args, criterion)
    else:
        if args.use_graph_emb:
            model = GraphBERT(config, args, criterion)
        else:
            if args.use_label:
                model = BertBiEncoderWithLabel(config, args, criterion)
            else:
                model = BertBiEncoder(config, args, criterion)


    if args.use_cuda:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'


    return model, test_loader, device


def save_result(result_dic, tmp_result_dir):
    
    os.makedirs(tmp_result_dir, exist_ok=True)

    for filename, result_jd in result_dic.items():
        output_filename = filename.replace(".json", ".rmd.json")
        tmp_dir = tmp_result_dir + output_filename
        tmp_dir = tmp_dir.split('/')[:-1]
        if not os.path.isdir("/".join(tmp_dir)):
            os.makedirs("/".join(tmp_dir))

        with open(tmp_result_dir + output_filename, mode="w") as out_f:
            json.dump(result_jd, out_f, indent=4)


def test_one_encoder(
    args,
    tmp_output_dir: str,
    result_dir: str,
):

    model, test_loader, device = pre_process(args, tmp_output_dir, result_dir)


    model = model.to(device)

    load_dir = glob.glob(tmp_output_dir+"checkpoint-*")[0]
    result_dic = {}
    model.load_state_dict(torch.load(load_dir+"/pytorch_model.bin"))
    for i, b in enumerate(test_loader):

        with torch.no_grad():
            input_ids = b["dialogue"]["input_ids"].to(device)
            token_type_ids= b["dialogue"]["token_type_ids"].to(device)
            attention_mask = b["dialogue"]["attention_mask"].to(device)

            if args.use_label:
                labels = b["label"].to(device)


            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                target_score=torch.tensor(b["score"]).to(device),
            )

        filename_list = [data["filename"] for data in b["info"]]
        speaker_list = [data["speaker"] for data in b["info"]]
        id_list = [data["id"] for data in b["info"]]

        for filename, speaker, spot_id, output in zip(filename_list, speaker_list, id_list, outputs.logits):
            if not filename in result_dic:
                result_dic[filename] = {}
            if not speaker in result_dic[filename]:
                result_dic[filename][speaker] = []

            result_dic[filename][speaker].append({
                "id": spot_id,
                "score": output.item()
            })

    save_result(result_dic, result_dir)




def test_double_encoder(
    args,
    tmp_output_dir: str,
    result_dir: str,
):

    model, test_loader, device = pre_process(args, tmp_output_dir, result_dir)


    model = model.to(device)

    load_dir = glob.glob(tmp_output_dir+"checkpoint-*")[0]
    result_dic = {}
    model.load_state_dict(torch.load(load_dir+"/pytorch_model.bin"))
    for i, b in enumerate(test_loader):

        with torch.no_grad():
            if args.use_label:
                labels = b["label"].to(device)

            dialogue_input_ids = b["dialogue"]["input_ids"].to(device)
            dialogue_token_type_ids= b["dialogue"]["token_type_ids"].to(device)
            dialogue_attention_mask = b["dialogue"]["attention_mask"].to(device)

            desc_input_ids = b["desc"]["input_ids"].to(device)
            desc_token_type_ids= b["desc"]["token_type_ids"].to(device)
            desc_attention_mask = b["desc"]["attention_mask"].to(device)

            outputs = model(
                dialogue_input_ids=dialogue_input_ids,
                dialogue_token_type_ids=dialogue_token_type_ids,
                dialogue_attention_mask=dialogue_attention_mask,
                desc_input_ids=desc_input_ids,
                desc_token_type_ids=desc_token_type_ids,
                desc_attention_mask=desc_attention_mask,
                target_score=torch.tensor(b["score"]).to(device),
            )

        filename_list = [data["filename"] for data in b["info"]]
        speaker_list = [data["speaker"] for data in b["info"]]
        id_list = [data["id"] for data in b["info"]]

        for filename, speaker, spot_id, output in zip(filename_list, speaker_list, id_list, outputs.logits):
            if not filename in result_dic:
                result_dic[filename] = {}
            if not speaker in result_dic[filename]:
                result_dic[filename][speaker] = []

            result_dic[filename][speaker].append({
                "id": spot_id,
                "score": output.item()
            })

    save_result(result_dic, result_dir)

