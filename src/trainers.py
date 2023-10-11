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
from utils.input_data import collate_fn
from utils.others import stack_data
from transformers import Trainer
from transformers import EarlyStoppingCallback
from transformers import logging
from transformers import BertJapaneseTokenizer

logging.set_verbosity_warning()

import numpy as np

def train_valid(
    args,
    output_dir: str,
    result_dir: str,
):

    print("*"*70)

    print(f"Split ID: {args.split_id}")
    print(f"Split File: {args.split_info_dir}")
    print(f"Dialogue Max Length: {args.dialogue_max_length}")
    print(f"Description Max Length: {args.desc_max_length}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"PLM Vocab Dir: {args.plm_vocab_dir}")

    print("*"*70)

    # dataset, dataloader
    train_dataset = MyDataset(args, exp_type="train")
    valid_dataset = MyDataset(args, exp_type="valid")
    test_dataset = MyDataset(args, exp_type="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


    # model
    config = transformers.AutoConfig.from_pretrained(args.pretrained_model_name)

    criterion = nn.MSELoss(reduction="mean")

    model = DoubleBERTEncoder(config, args, criterion)

    if args.use_cuda:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'


    model = model.to(device)


    for try_num in range(args.max_try):
        tmp_output_dir = output_dir+f"try_{try_num}/"

        training_args = TrainingArguments(
            output_dir=tmp_output_dir,
            evaluation_strategy='steps',
            logging_strategy='steps',
            logging_steps=100,
            save_steps=100,
            save_total_limit=2,
            warmup_steps=args.warmup_steps,
            label_names=['target_score'],
            learning_rate=args.lr,
            metric_for_best_model='loss',
            load_best_model_at_end=True,
            per_device_train_batch_size=int(args.train_batch_size/args.cuda_num),
            per_device_eval_batch_size=int(args.train_batch_size/args.cuda_num),
            num_train_epochs=args.epochs,
            remove_unused_columns=False,
            optim=args.optimizer,
        )

        collater = BertBiEncoderCollator(args)

        trainer = Trainer(
            model=model,
            data_collator=collater,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )

        # Train
        trainer.train()
        pred_results = trainer.predict(test_dataset)
        tmp_result_dir = result_dir + f"try_{try_num}/"
        file_result_dict = {}

        # Test & Save
        for pred, data in zip(pred_results.predictions, test_dataset):
            info = data[-1]
            if info["filename"] not in file_result_dict.keys():
                file_result_dict[info["filename"]] = {}
            if info["speaker"] not in file_result_dict[info["filename"]].keys():
                file_result_dict[info["filename"]][info["speaker"]] = []

            file_result_dict[info["filename"]][info["speaker"]].append({
                "id": info["id"],
                "score": float(pred)
            })

        for filename in file_result_dict.keys():
            os.makedirs(tmp_result_dir, exist_ok=True)
            with open(tmp_result_dir+filename.replace(".json", ".rmd.json"), "w", encoding='utf-8') as f:
                json.dump(file_result_dict[filename], f, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained_model_name", help="PLMの名前")

    # Dataset
    parser.add_argument("--data_type", help="データの種類")
    parser.add_argument("--split_info_dir", help="")
    parser.add_argument("--data_dir", help="対話&評価のjsonファイルのディレクトリ")
    parser.add_argument("--dialogue_max_length", type=int, default=512, help="モデルに入力する対話履歴の最大トークン長")
    parser.add_argument("--desc_max_length", type=int, default=512, help="モデルに入力する観光地説明文の最大トークン長")
    parser.add_argument("--use_summary", action='store_true')
    parser.add_argument("--use_summary_turn_5", action='store_true')
    parser.add_argument("--rec_sentence", action='store_true')

    # Others
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--cuda_num", type=int)
    parser.add_argument("--split_id_list", type=str)
    parser.add_argument("--split_id", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--max_try", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int)


    args = parser.parse_args()

    output_dir = './outputs/model/'
    result_dir = './outputs/result/'

    if args.rec_sentence:
        if args.use_summary_turn_5:
            args.data_dir += 'sum_rec_chat_and_rec_turn_5/'
            output_dir += '-summary-rec-turn_5'
            result_dir += '-summary-rec-turn_5'
        elif args.use_summary:
            args.data_dir += 'sum_rec_chat_and_rec/'
            output_dir += '-summary-rec'
            result_dir += '-summary-rec'
        else:
            args.data_dir += 'sum_rec_chat_and_rec/'
            output_dir += '-rec'
            result_dir += '-rec'
    elif args.use_summary:
        args.data_dir += 'sum_chat_and_rec/'
        output_dir += '-summary'
        result_dir += '-summary'
    elif args.use_summary_turn_5:
        args.data_dir += 'sum_chat_and_rec_turn_5/'
        output_dir += '-summary_turn_5'
        result_dir += '-summary_turn_5'
    else:
        args.data_dir += 'chat_and_rec/'

    train_valid(args, output_dir, result_dir)

