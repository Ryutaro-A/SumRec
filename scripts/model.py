from transformers import BertModel, AutoModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.outputs import BiEncoderOutput, AttentionBiEncoderOutput
from transformers.modeling_outputs import ModelOutput


class BaseOneEncoder(nn.Module):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):

        super().__init__()
        if args.use_pretrain_model:
            self.bert = AutoModel.from_pretrained(
                args.pretrained_model_name,
                config,
                output_attentions=True,
                return_dict=True,
            )
        else:
            self.bert = AutoModel(config)


class OneEncoder(BaseOneEncoder):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):
        super().__init__(config, args, loss_function)

        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_function = loss_function

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor=None,
        target_score: torch.Tensor=None,
    ):

        out = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        out = self.linear(out.pooler_output)

        out = out.squeeze()

        # print(out.size())
        # print(target_score.size())
        # exit()

        loss = self.loss_function(out, target_score)


        return ModelOutput(
            logits=out,
            loss=loss,
        )



class BaseBiEncoder(nn.Module):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):

        super().__init__()
        if args.use_pretrain_model:
            self.bert = AutoModel.from_pretrained(
                args.pretrained_model_name,
                config,
                output_attentions=True,
                return_dict=True,
            )
        else:
            self.bert = AutoModel(config)


class BaseDoubleBERTEncoder(nn.Module):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):

        super().__init__()
        if args.use_pretrain_model:
            # 対話履歴用のEncoder
            self.speaker_bert = AutoModel.from_pretrained(
                args.pretrained_model_name,
                config,
                output_attentions=True,
                return_dict=True,
            )

            # 観光地説明文用のEncoder
            self.spot_bert = AutoModel.from_pretrained(
                args.pretrained_model_name,
                config,
                output_attentions=True,
                return_dict=True,
            )
        else:
            self.peaker_bert = AutoModel(config)
            self.spot_bert = AutoModel(config)



class DoubleBERTEncoder(BaseDoubleBERTEncoder):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):
        super().__init__(config, args, loss_function)

        self.linear = nn.Linear(config.hidden_size*2, 1)
        self.loss_function = loss_function

    def forward(self,
        speaker_input_ids: torch.Tensor,
        speaker_token_type_ids: torch.Tensor,
        speaker_attention_mask: torch.Tensor=None,
        spot_input_ids: torch.Tensor=None,
        spot_token_type_ids: torch.Tensor=None,
        spot_attention_mask: torch.Tensor=None,
        target_score: torch.Tensor=None,
    ):

        speaker = self.speaker_bert(
            input_ids=speaker_input_ids,
            token_type_ids=speaker_token_type_ids,
            attention_mask=speaker_attention_mask,
        )

        spot = self.spot_bert(
            input_ids=spot_input_ids,
            token_type_ids=spot_token_type_ids,
            attention_mask=spot_attention_mask,
        )

        out = torch.cat([speaker.pooler_output, spot.pooler_output], dim=1)

        out = self.linear(out)

        out = out.squeeze()

        loss = self.loss_function(out, target_score)

        # attentionをreturnするようにすると学習時間が３倍くらいになる！
        # dialogue_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(dialogue.attentions))]
        # desc_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(desc.attentions))]

        # dialogue_attentions = [data.detach() for data in dialogue.attentions]
        # desc_attentions = [data.detach() for data in desc.attentions]

        return ModelOutput(
            logits=out,
            loss=loss,
            # dialogue_attentions=dialogue_attentions,
            # desc_attentions=desc_attentions,
        )


class DoubleBERTEncoderWithLabel(BaseDoubleBERTEncoder):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):
        super().__init__(config, args, loss_function)

        self.linear = nn.Linear(config.hidden_size*2, 1)
        self.loss_function = loss_function

        self.type_embedings = nn.Embedding(2, config.hidden_size)
        self.bert_embedings = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self,
        dialogue_input_ids: torch.Tensor,
        dialogue_token_type_ids: torch.Tensor,
        dialogue_attention_mask: torch.Tensor=None,
        desc_input_ids: torch.Tensor=None,
        desc_token_type_ids: torch.Tensor=None,
        desc_attention_mask: torch.Tensor=None,
        target_score: torch.Tensor=None,
        labels: torch.Tensor=None,
    ):

        # print(labels)
        # exit()

        label_embeding = self.type_embedings(labels)
        input_embeding = self.bert_embedings(dialogue_input_ids)
        # token_type_embeding = self.token_type_embedings(input_dic["token_type_ids"])
        inputs_embeds = input_embeding + label_embeding # + token_type_embeding

        dialogue = self.dialogue_bert(
            inputs_embeds=inputs_embeds,
            token_type_ids=dialogue_token_type_ids,
            attention_mask=dialogue_attention_mask,
        )

        desc = self.desc_bert(
            input_ids=desc_input_ids,
            token_type_ids=desc_token_type_ids,
            attention_mask=desc_attention_mask,
        )

        out = torch.cat([dialogue.pooler_output, desc.pooler_output], dim=1)

        out = self.linear(out)

        out = out.squeeze()

        loss = self.loss_function(out, target_score)

        # attentionをreturnするようにすると学習時間が３倍くらいになる！
        # dialogue_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(dialogue.attentions))]
        # desc_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(desc.attentions))]

        # dialogue_attentions = [data.detach() for data in dialogue.attentions]
        # desc_attentions = [data.detach() for data in desc.attentions]

        return ModelOutput(
            logits=out,
            loss=loss,
            # dialogue_attentions=dialogue_attentions,
            # desc_attentions=desc_attentions,
        )


class BertBiEncoder(BaseBiEncoder):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):
        super().__init__(config, args, loss_function)

        self.linear = nn.Linear(config.hidden_size*2, 1)
        self.loss_function = loss_function

    def forward(self,
        dialogue_input_ids: torch.Tensor,
        dialogue_token_type_ids: torch.Tensor,
        dialogue_attention_mask: torch.Tensor=None,
        desc_input_ids: torch.Tensor=None,
        desc_token_type_ids: torch.Tensor=None,
        desc_attention_mask: torch.Tensor=None,
        target_score: torch.Tensor=None,
    ):

        dialogue = self.bert(
            input_ids=dialogue_input_ids,
            token_type_ids=dialogue_token_type_ids,
            attention_mask=dialogue_attention_mask,
        )

        desc = self.bert(
            input_ids=desc_input_ids,
            token_type_ids=desc_token_type_ids,
            attention_mask=desc_attention_mask,
        )

        out = torch.cat([dialogue.pooler_output, desc.pooler_output], dim=1)

        out = self.linear(out)

        out = out.squeeze()

        loss = self.loss_function(out, target_score)

        # attentionをreturnするようにすると学習時間が３倍くらいになる！
        # dialogue_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(dialogue.attentions))]
        # desc_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(desc.attentions))]

        # dialogue_attentions = [data.detach() for data in dialogue.attentions]
        # desc_attentions = [data.detach() for data in desc.attentions]

        return ModelOutput(
            logits=out,
            loss=loss,
            # dialogue_attentions=dialogue_attentions,
            # desc_attentions=desc_attentions,
        )

class BertBiEncoderWithLabel(BaseBiEncoder):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):
        super().__init__(config, args, loss_function)

        self.linear = nn.Linear(config.hidden_size*2, 1)
        self.use_label = args.use_label
        self.loss_function = loss_function

        self.type_embedings = nn.Embedding(2, config.hidden_size)
        self.bert_embedings = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.token_type_embedings = nn.Embedding(2, 768)

    def forward(self,
        dialogue_input_ids: torch.Tensor,
        dialogue_token_type_ids: torch.Tensor,
        dialogue_attention_mask: torch.Tensor=None,
        desc_input_ids: torch.Tensor=None,
        desc_token_type_ids: torch.Tensor=None,
        desc_attention_mask: torch.Tensor=None,
        target_score: torch.Tensor=None,
        labels: torch.Tensor=None,
    ):

        label_embeding = self.type_embedings(labels)
        input_embeding = self.bert_embedings(dialogue_input_ids)
        # token_type_embeding = self.token_type_embedings(input_dic["token_type_ids"])
        inputs_embeds = input_embeding + label_embeding # + token_type_embeding
        dialogue = self.bert(
            inputs_embeds=inputs_embeds,
            token_type_ids=dialogue_token_type_ids,
            attention_mask=dialogue_attention_mask,
        )

        desc = self.bert(
            input_ids=desc_input_ids,
            token_type_ids=desc_token_type_ids,
            attention_mask=desc_attention_mask,
        )

        out = torch.cat([dialogue.pooler_output, desc.pooler_output], dim=1)

        out = self.linear(out)

        out = out.squeeze()

        loss = self.loss_function(out, target_score)

        # attentionをreturnするようにすると学習時間が３倍くらいになる！
        # dialogue_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(dialogue.attentions))]
        # desc_attentions = [dialogue.attentions[i].detach().cpu() for i in range(len(desc.attentions))]

        # dialogue_attentions = [data.detach() for data in dialogue.attentions]
        # desc_attentions = [data.detach() for data in desc.attentions]

        return ModelOutput(
            logits=out,
            loss=loss,
            # dialogue_attentions=dialogue_attentions,
            # desc_attentions=desc_attentions,
        )




class GraphLinearBERT(BaseBiEncoder):
    def __init__(
        self,
        config,
        args,
    ):
        super().__init__(config, args)

        linear_size = args.graph_emb_dim*args.max_entity + config.hidden_size*2

        # classify layer
        self.linear = nn.Linear(linear_size, 1)

        self.use_label = args.use_label
        if self.use_label:
            self.type_embedings = nn.Embedding(2, config.hidden_size)
            self.bert_embedings = nn.Embedding(config.vocab_size, config.hidden_size)
            # self.token_type_embedings = nn.Embedding(2, 768)

    def forward(self,
        dialogue_input_ids: torch.Tensor=None,
        dialogue_token_type_ids: torch.Tensor=None,
        dialogue_attention_mask: torch.Tensor=None,
        desc_input_ids: torch.Tensor=None,
        desc_token_type_ids: torch.Tensor=None,
        desc_attention_mask: torch.Tensor=None,
        labels: torch.Tensor=None,
        graph_emb: torch.Tensor=None,
        exp_type: str='train',
    ):

        if self.use_label:
            # print(input_dic["input_ids"])
            label_embeding = self.type_embedings(labels)
            input_embeding = self.bert_embedings(dialogue_input_ids)
            # token_type_embeding = self.token_type_embedings(input_dic["token_type_ids"])
            inputs_embeds = input_embeding + label_embeding # + token_type_embeding
            dialogue = self.bert(
                inputs_embeds=inputs_embeds,
                token_type_ids=dialogue_token_type_ids,
                # attention_mask=dialogue_attention_mask,
            )

            desc = self.bert(
                input_ids=desc_input_ids,
                token_type_ids=desc_token_type_ids,
                attention_mask=desc_attention_mask,
            )
        else:
            dialogue = self.bert(
                input_ids=dialogue_input_ids,
                token_type_ids=dialogue_token_type_ids,
                attention_mask=dialogue_attention_mask,
            )

            desc = self.bert(
                input_ids=desc_input_ids,
                token_type_ids=desc_token_type_ids,
                attention_mask=desc_attention_mask,
            )


        out = torch.cat([graph_emb, dialogue.pooler_output, desc.pooler_output], dim=1)

        out = self.linear(out)

        out = out.squeeze()

        loss = self.loss_function(out, target_score)
        # dialogue_attention = [data.detach() for data in dialogue.attentions]
        # desc_attention = [data.detach() for data in desc.attentions]

        return out

        # return AttentionBiEncoderOutput(
        #     output=out,
        #     dialogue_attentions=dialogue_attention,
        #     desc_attentions=desc_attention,
        #     cross_attentions=attn_weights
        # )




class GraphBERT(BaseBiEncoder):
    def __init__(
        self,
        config,
        args,
        loss_function,
    ):
        super().__init__(config, args, loss_function)

        linear_size = config.hidden_size*2

        # classify layer
        self.linear = nn.Linear(linear_size, 1)

        self.loss_function = loss_function

        # 対話にも観光地説明文にもentityのトークンにもこのembedingsを使う
        self.bert_embedings = nn.Embedding(config.vocab_size, config.hidden_size)
        # 観光地説明文かentityかを区別する0 or 1の埋め込み
        self.token_type_embedings = nn.Embedding(2, config.hidden_size)

    def forward(self,
        dialogue_input_ids: torch.Tensor=None,
        dialogue_token_type_ids: torch.Tensor=None,
        dialogue_attention_mask: torch.Tensor=None,
        desc_input_ids: torch.Tensor=None,
        desc_token_type_ids: torch.Tensor=None,
        desc_attention_mask: torch.Tensor=None,
        graph_emb: torch.Tensor=None,
        target_score: torch.Tensor=None,
    ):


        dialogue = self.bert(
            input_ids=dialogue_input_ids,
            token_type_ids=dialogue_token_type_ids,
            attention_mask=dialogue_attention_mask,
        )

        desc_embeding = self.bert_embedings(desc_input_ids)
        desc_graph_embeding = desc_embeding + graph_emb

        desc = self.bert(
            inputs_embeds=desc_graph_embeding,
            token_type_ids=desc_token_type_ids,
        )


        out = torch.cat([dialogue.pooler_output, desc.pooler_output], dim=1)

        out = self.linear(out)

        out = out.squeeze()

        loss = self.loss_function(out, target_score)
        # dialogue_attention = [data.detach() for data in dialogue.attentions]
        # desc_attention = [data.detach() for data in desc.attentions]

        return ModelOutput(
            logits=out,
            loss=loss,
            # dialogue_attentions=dialogue_attentions,
            # desc_attentions=desc_attentions,
        )