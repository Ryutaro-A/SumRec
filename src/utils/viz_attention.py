import itertools
from operator import itemgetter
import torch
from transformers import AutoTokenizer
import numpy as np
import os


# 赤くハイライトする
def highlight_r(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

# 青くハイライトする
def highlight_b(word, attn):
    html_color = '#%02X%02X%02X' % (int(255*(1 - attn)), int(255*(1 - attn)), 255)
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def show_bert_attention_backup(dataset):
    for idx, b in enumerate(dataset):

        input_ids = b["input_ids"].to(device).unsqueeze(0)
        attention_mask = b["attention_mask"].to(device).unsqueeze(0)
        token_type_ids = b["token_type_ids"].to(device).unsqueeze(0)
        target = b["target"].to(device)

        with torch.no_grad():
            output, attention = model(input_ids, attention_mask, token_type_ids)

        attention = attention.cpu()[0].numpy()
        attention_mask = attention_mask.cpu()[0].numpy()
        attention = attention[attention_mask == 1][1:-1]

        ids = input_ids.cpu()[0][attention_mask == 1][1:-1].tolist()
        tokens = TOKENIZER.convert_ids_to_tokens(ids)

        html_outputs = []

        for word, attn in zip(tokens, attention):
            html_outputs.append(highlight_r(word, attn))

def mark_attention(attention, input_ids, attention_mask, tokenizer):

    # attention = attention.numpy()
    attention_mask = attention_mask.numpy()
    attention = attention[attention_mask == 1][1:-1]

    ids = input_ids[attention_mask == 1][1:-1].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    html_outputs = []

    for word, attn in zip(tokens, attention):
        word = word.replace("#", "")
        if len(word) == 0:
            continue
        html_outputs.append(highlight_r(word, attn))

    return "".join(html_outputs)

def show_bert_attention(input_dic_list, speakers, tokenizer):
    # speakerごとのkength2のリストを送る
    result = []

    # for i, input_dic in enumerate(input_dic_list):
    #     if i == 0:
    #         desc_html = "<br><br>".join([mark_attention(d["desc_attention"], d["desc_input_ids"], d["desc_attention_mask"], tokenizer) for d in input_dic])
    #         result.append({
    #             "file_type": "place",
    #             "html": "".join(desc_html),
    #         })
    #     dialogue_html = "<br><br>".join([mark_attention(d["dialogue_attention"], d["dialogue_input_ids"], d["dialogue_attention_mask"], tokenizer) for d in input_dic])
    #     # dialogue_html = mark_attention([data["dialogue_attention"] for data in input_dic], [data["dialogue_input_ids"] for data in input_dic], [data["dialogue_attention_mask"] for data in input_dic], tokenizer)
    #     result.append({
    #         "file_type": "dialogue",
    #         "html": "".join(dialogue_html),
    #         "speaker": speakers[i],
    #     })

    for i, input_dic in enumerate(input_dic_list):
        if i == 0:
            dialogue_html = mark_attention(input_dic[0]["dialogue_attention"], input_dic[0]["dialogue_input_ids"], input_dic[0]["dialogue_attention_mask"], tokenizer)
            result.append(dialogue_html)

        desc_html = "<br><br>".join([mark_attention(d["desc_attention"], d["desc_input_ids"], d["desc_attention_mask"], tokenizer) for d in input_dic])
        result.append(desc_html)

    return result