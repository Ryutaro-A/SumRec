import re
import os
import random
import sys
import json
from copy import deepcopy
from collections import OrderedDict
import polars as pl
import itertools
import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertJapaneseTokenizer

from utils.input_data import get_token, collate_fn
tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-large-japanese-seq512-with-auto-jumanpp")

class MyDataset(Dataset):

    def __init__(
        self,
        args,
        exp_type,
    ):


        if args.unseen:
            split_info_file = f'{args.split_info_dir}unseen/{args.data_type}_split.json'
        else:
            split_info_file = f'{args.split_info_dir}seen/{args.data_type}_split.json'
        with open(split_info_file, encoding='utf-8') as f:
            split_info_jd = json.load(f, object_pairs_hook=OrderedDict)
        filename_list = split_info_jd[args.split_id][exp_type]


        self.noise_method = args.noise_method
        self.use_label = args.use_label
        self.use_graph_emb = args.use_graph_emb
        self.use_entity = args.use_entity
        self.use_entity_bert = args.use_entity_bert
        self.use_need_utt = args.use_need_utt
        self.use_summary = args.use_summary
        self.use_summary_turn_5 = args.use_summary_turn_5
        self.one_encoder = args.one_encoder
        self.pretrained_model_name = args.pretrained_model_name
        self.norm_dataset = args.norm_dataset
        self.rec_sentence = args.rec_sentence

        if self.use_entity_bert or self.use_graph_emb:
            self.knowledge = pl.read_csv(args.knowledge_path)
            self.all_entities = self.knowledge["nodes"].unique().to_list()

        
        if self.use_summary or self.use_summary_turn_5:
            self.max_speaker_length = 210
        else:
            self.max_speaker_length = 512 # 対話履歴全部入れると512を軽く超える

        if self.rec_sentence:
            self.max_spot_length = 250
        else:
            # 観光地説明文の最大長に合わせる
            self.max_spot_length = 216

        self.max_sum_token = 0


        self.tag_dic = {
            "gratitude": "0",
            "repeat": "1",
            "acknowledge": "2",
            "self": "3",
            "paraphrase": "4",
            "suggestion": "5",
            "wonder": "6",
            "inform": "7",
            "filler": "8",
            "sympathy": "9",
            "question": "10",
            "greeting": "11",
            "noempathy": "12",
            "apology": "13",
            "confirm": "14",
            "accept": "15",
        }

        self.need_list = ["3", "7", "5"]

        self.data_type = args.data_type

        self.graph_emb_dim = args.graph_emb_dim
        self.max_entity = args.max_entity
        self.dialogue_max_length = args.dialogue_max_length
        self.desc_max_length = args.desc_max_length

        if "roberta" in args.plm_vocab_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(args.plm_vocab_dir)
        else:
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.plm_vocab_dir)
            
        self.pad_id = self.tokenizer.pad_token_id

        self.turn_num = args.turn_num

        self.data_list = self.get_data_list(args.data_dir, filename_list)
        self.data_list = sum(self.data_list, [])

        self.len = len(self.data_list)



    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        dialogue = self.data_list[index]["speaker"]
        desc = self.data_list[index]["spot"]
        score = self.data_list[index]["score"]
        info = self.data_list[index]["info"]

        return dialogue, desc, score, info



    def score_scaling(self, score, down_scale=False):
        if not down_scale:
            return (score - 3) / 2.0
        if score == 2:
            return self.score_scaling(1)
        elif score == 4:
            return self.score_scaling(5)
        else:
            return self.score_scaling(score)

    def token_padding(
        self,
        encoded_dict: dict,
        token_type: str,
    ):
        if token_type == "desc_rec":
            max_len = self.max_spot_length
        elif token_type == "dialogue":
            max_len = self.dialogue_max_length

        for k in encoded_dict.keys():
            src_length = len(encoded_dict[k])
            if src_length < max_len:
                tmp_obj = torch.tensor(encoded_dict[k][:])
                # padding: input_idsはPAD_ID埋め，token_type_idsは0埋め，attention_maskは[PAD]の箇所0
                packed_obj = pack_sequence([tmp_obj])
                padding_id = self.pad_id if k == "input_ids" else 0
                padded_src = pad_packed_sequence(packed_obj, batch_first=True, padding_value=padding_id, total_length=max_len)
                encoded_dict[k] = padded_src[0][0]
            elif src_length > max_len:
                encoded_dict[k] = torch.tensor(deepcopy(encoded_dict[k][:max_len]))
            else:
                encoded_dict[k] = torch.tensor(deepcopy(encoded_dict[k]))


            if len(encoded_dict[k]) != max_len:
                print("not fit length!", len(encoded_dict[k]))

        return encoded_dict



    def get_speaker_dic_rec(self, rec_sentences):
        result_dic = {}
        for i, rec in enumerate(rec_sentences):
            encode_dict = self.tokenizer(
                rec,
                add_special_tokens=True,
            )


            result_dic[str(i+1)] = encode_dict
        return result_dic


    def get_speaker_dic(self, d_list, speakers):

        dialogue_dic = {}
        if self.use_label:
            label_dic = {}
        if self.use_graph_emb:
            graph_dic = {}
        if self.use_entity_bert or self.use_graph_emb:
            entity_bert_dic = {}

        end_flag = False
        for speaker in speakers:
            entity_count = 0
            entity_flag = False # max_entityに達したら終了するためのフラグ
            for i, utt_dict in enumerate(d_list):

                # ターン数がturn_numを超えたらそこで終了
                if self.turn_num != -1 and i / 2 == self.turn_num:
                    # end_flag = True
                    break

                # 予測する話者のtoken_type_idsを1にする
                if utt_dict["speaker"] == speaker:
                    target_token = 1
                else:
                    target_token = 0

                encode_dict = self.tokenizer(utt_dict["utterance"], add_special_tokens=True)
                encode_dict["token_type_ids"] = [target_token]*len(encode_dict["input_ids"])


                if speaker in dialogue_dic.keys():
                    for k in encode_dict.keys():
                        dialogue_dic[speaker][k].extend(encode_dict[k][1:])
                else:
                    for k in encode_dict.keys():
                        dialogue_dic[speaker] = encode_dict

            dialogue_dic[speaker] = deepcopy(dialogue_dic[speaker])

            # length = len(dialogue_dic[speaker]["input_ids"])
            # if length > self.max_sum_token:
            #     self.max_sum_token = length

            dialogue_dic[speaker] = self.token_padding(dialogue_dic[speaker], token_type="dialogue")


        return dialogue_dic


    def get_speaker_dic_summary(self, chat_rec_jd, speakers):
        result_dic = {}
        for speaker in speakers:

            # encode_dict = self.tokenizer(chat_rec_jd["summary"][speaker], add_special_tokens=True)
            # if self.max_sum_token < len(encode_dict["input_ids"]):
            #     self.max_sum_token = len(encode_dict["input_ids"])

            encode_dict = self.tokenizer(
                chat_rec_jd["summary"][speaker],
                # return_tensors='pt',
                add_special_tokens=True,
                max_length=self.max_speaker_length,
                truncation=True,
                padding="max_length"
            )

            result_dic[speaker] = {}

            for k in list(encode_dict.keys()):
                result_dic[speaker][k] = torch.tensor(encode_dict[k])
            

        return result_dic


    # 発話埋め込みにラベル埋め込みを加算してinput_dicを返す
    def get_input_embeds(
        self,
        input_dic: dict,
        label_dic: dict,
    ):
        label_dic = deepcopy(label_dic)
        input_dic = deepcopy(input_dic)
        # print(input_dic["input_ids"])
        label_embeding = self.type_embedings(label_dic["input_ids"])
        input_embeding = self.bert_embedings(input_dic["input_ids"])
        # token_type_embeding = self.token_type_embedings(input_dic["token_type_ids"])
        input_dic["input_ids"] = input_embeding + label_embeding # + token_type_embeding
        return input_dic

    def concat_desc_entity(self, desc_dict, entity_dict):
        con = {
            "input_ids": desc_dict["input_ids"] + entity_dict["input_ids"],
            "token_type_ids": desc_dict["token_type_ids"] + [1]*len(entity_dict["token_type_ids"]),
            "attention_mask": desc_dict["attention_mask"] + entity_dict["attention_mask"],
        }
        return self.token_padding(con, token_type="desc")

    # 話者ごとに入れ子になったdictのlistを返す
    def get_data_list(self, data_dir, filename_list):
        data_list = []
        b = []
        for filename in filename_list:
            with open(data_dir+filename) as f:
                chat_rec_jd = json.load(f, object_pairs_hook=OrderedDict)

            speakers = list(chat_rec_jd["questionnaire"].keys())

            if self.use_summary or self.use_summary_turn_5:
                speaker_dic = self.get_speaker_dic_summary(chat_rec_jd, speakers)
            else:
                speaker_dic = self.get_speaker_dic(chat_rec_jd["dialogue"], speakers)


            # key: 観光地ID, value: トークナイズ後の観光地説明文
            desc_dic = {}
            for place_dict in chat_rec_jd["place"]:
                input_desc = place_dict["description"]
                if self.rec_sentence:
                    encode_dict = self.tokenizer(
                        input_desc,
                        add_special_tokens=True,
                        max_length=self.dialogue_max_length,
                        truncation=True,
                    )
                    # for speaker in speakers:
                    #     spot_length = len(encode_dict["input_ids"]) + len(speaker_dic[speaker]["input_ids"])
                    #     if self.max_sum_token < len(encode_dict["input_ids"]):
                    #         self.max_sum_token = len(encode_dict["input_ids"])
                else:
                    # encode_dict = self.tokenizer(input_desc, add_special_tokens=True)
                    # if self.max_sum_token < len(encode_dict["input_ids"]):
                    #     self.max_sum_token = len(encode_dict["input_ids"])
                    encode_dict = self.tokenizer(
                        input_desc,
                        # return_tensors='pt',
                        add_special_tokens=True,
                        max_length=self.max_spot_length,
                        truncation=True,
                        padding="max_length"
                    )
                    for k in list(encode_dict.keys()):
                        encode_dict[k] = torch.tensor(encode_dict[k])
                for k in encode_dict.keys():
                    desc_dic[place_dict["id"]] = encode_dict
            
            if self.rec_sentence:
                rec_dic = self.get_speaker_dic_rec([data["recommend"] for data in chat_rec_jd["place"]])


            for speaker in speakers:
                tmp_list = []
                # print(self.tokenizer.decode(speaker_dic[speaker]["input_ids"][0]))
                # print(len(speaker_dic[speaker]["input_ids"][0]))
                # 対話履歴，観光地説明文，観光地ID，観光地へのスコアの1つ分の辞書を作成
                for eval_dict in chat_rec_jd["questionnaire"][speaker]["evaluation"]:
                    if self.rec_sentence:
                        # rec_dic = self.get_speaker_dic_rec([data["recommend"] for data in chat_rec_jd["place"]])
                        spot_dic = {}
                        for k in speaker_dic[speaker].keys():
                            spot_dic[k] = desc_dic[eval_dict["id"]][k] + rec_dic[eval_dict["id"]][k][1:]
                        spot_dic = self.token_padding(spot_dic, token_type="desc_rec")
                        # print(self.tokenizer.decode(spot_dic["input_ids"]))
                        # print(len(spot_dic["input_ids"]))
                    else:
                        spot_dic = desc_dic[eval_dict["id"]]

                    # print(speaker_dic[speaker]["input_ids"][0])
                    

                    data_dic = {
                        "speaker": speaker_dic[speaker],
                        "spot": spot_dic,
                        "score": self.score_scaling(eval_dict["score"]) if not self.norm_dataset else eval_dict["score"],
                        "info": {
                            "id": eval_dict["id"],
                            "filename": os.path.basename(filename),
                            "speaker": speaker,
                        }
                    }


                    tmp_list.append(data_dic)

                data_list.append(tmp_list)

        print(self.max_sum_token)

        return data_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained_model_name", help="PLMの名前")
    parser.add_argument("--use_pretrain_model", action='store_true')
    parser.add_argument("--use_large_model", action='store_true')
    parser.add_argument("--plm_hidden_dropout", type=float, default=0.0)
    parser.add_argument("--plm_attention_dropout", type=float, default=0.0)
    parser.add_argument("--use_double_encoder", action='store_true')

    # Dataset
    parser.add_argument("--plm_vocab_dir", help="", type=str, default="")
    parser.add_argument("--data_type", help="データの種類")
    parser.add_argument("--split_info_dir", help="")
    parser.add_argument("--data_dir", help="対話&評価のjsonファイルのディレクトリ")
    parser.add_argument("--unseen", action='store_true', help="テストファイルをunseenとするか")
    parser.add_argument("--dialogue_max_length", type=int, default=512, help="モデルに入力する対話履歴の最大トークン長")
    parser.add_argument("--desc_max_length", type=int, default=512, help="モデルに入力する観光地説明文の最大トークン長")
    parser.add_argument("--noise_method", type=str, default=None, help="dialogue_shuffle: 発話単位でシャッフル, utt_shuffle: 対話履歴は崩さずに発話内のトークン単位でシャッフル")
    parser.add_argument("--turn_num", type=int, default=-1)
    parser.add_argument("--graph_emb_dim", type=int, default=-1)
    parser.add_argument("--knowledge_path", help="")
    parser.add_argument("--max_entity", type=int, default=-1)
    parser.add_argument("--use_label", action='store_true')
    parser.add_argument("--use_graph_emb", action='store_true')
    parser.add_argument("--use_entity_bert", action='store_true')
    parser.add_argument("--use_entity", action='store_true')
    parser.add_argument("--use_need_utt", action='store_true')
    parser.add_argument("--use_summary", action='store_true')
    parser.add_argument("--use_summary_turn_5", action='store_true')
    parser.add_argument("--one_encoder", action='store_true')
    parser.add_argument("--norm_dataset", action='store_true')
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

    valid_dataset = MyDataset(args, exp_type="valid")
    valid_dataset = MyDataset(args, exp_type="train")
    valid_dataset = MyDataset(args, exp_type="test")
    # print(valid_dataset.max_sum_token)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn)

    # ## 0が対話履歴，1が観光地説明文，2がスコア
    # for batch in valid_dataloader:
    #     print(batch)
    #     a = batch["desc"]["input_ids"][0]
    #     b = batch["graph"][0].size()
    #     print(b)
    #     a = valid_dataset.tokenizer.decode(a)
    #     print(a)
    #     break



def get_token(target):
    input_ids = [data["input_ids"] for data in target]
    token_type_ids = [data["token_type_ids"] for data in target]
    attention_mask = [data["attention_mask"] for data in target]
    return {
        "input_ids": torch.stack(input_ids),
        "token_type_ids": torch.stack(token_type_ids),
        "attention_mask": torch.stack(attention_mask)
    }



# TODO: 各手法ごとにdatacollaterを用意したほうがメモリ的に良さそう
class BertBiEncoderCollator():
    def __init__(self, args):
        self.use_label = args.use_label
        # self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.plm_vocab_dir)

    def __call__(self, examples):
        speaker, spot, score, info = list(zip(*examples))


        speaker_dict = get_token(speaker)
        speaker_dict = {'speaker_'+key: speaker_dict[key] for key in speaker_dict.keys()}
        spot_dict = get_token(spot)
        spot_dict = {'spot_'+key: spot_dict[key] for key in spot_dict.keys()}

        print("".join(tokenizer.convert_ids_to_tokens(speaker_dict["speaker_input_ids"][0])))
        print("".join(tokenizer.convert_ids_to_tokens(spot_dict["spot_input_ids"][0])))
        # exit()

        collated_data = {
            **speaker_dict,
            **spot_dict,
            "target_score": torch.tensor(score),
            # "info": info
        }


        return collated_data


class GraphBERTCollator():
    def __init__(self, args):
        self.use_label = args.use_label

    def __call__(self, examples):
        dialogue, desc, score, label, graph, info = list(zip(*examples))

        dialogue_dict = get_token(dialogue)
        dialogue_dict = {'dialogue_'+key: dialogue_dict[key] for key in dialogue_dict.keys()}
        desc_dict = get_token(desc)
        desc_dict = {'desc_'+key: desc_dict[key] for key in desc_dict.keys()}

        if self.use_label:
            collated_data = {
                **dialogue_dict,
                **desc_dict,
                "target_score": torch.tensor(score),
                "labels": torch.stack(label),
                "graph_emb": torch.tensor(graph),
                # "info": info
            }
        else:
            collated_data = {
                **dialogue_dict,
                **desc_dict,
                "target_score": torch.tensor(score),
                "graph_emb": torch.tensor(graph),
                # "info": info
            }


        return collated_data


class BertOneEncoderCollator():
    def __init__(self, args):
        self.use_label = args.use_label
        # self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.plm_vocab_dir)

    def __call__(self, examples):
        dialogue_desc, _, score, label, _, info = list(zip(*examples))


        input_dict = get_token(dialogue_desc)
        # input_dict = {key: dialogue_dict[key] for key in input_dict.keys()}

        print("".join(tokenizer.convert_ids_to_tokens(input_dict["input_ids"][0])))
        # print(tokenizer.decode(input_dict["input_ids"][0]))
        # exit()

        if self.use_label:
            collated_data = {
                **input_dict,
                "target_score": torch.tensor(score),
                "labels": torch.stack(label),
                # "info": info
            }
        else:
            collated_data = {
                **input_dict,
                "target_score": torch.tensor(score),
                # "info": info
            }


        return collated_data