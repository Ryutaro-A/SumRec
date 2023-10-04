TYPE=travel
SPLIT_ID=2
METHOD=w2v_svr
python scripts/run.py \
    --data_type $TYPE \
    --method $METHOD \
    --data_dir ./data/chat_and_rec/ \
    --split_info_dir ./data/crossval_split/ \
    --split_id $SPLIT_ID \
    --model_output_dir ./saved_model/$METHOD/$TYPE\_split/$SPLIT_ID/ \
    --output_dir ./result/$METHOD/$TYPE\_split/$SPLIT_ID/ \
    --vocab_dir ./pretrained-transformer/roberta/roberta_dic/ \
    --mecab_dict ./mecab/mecab-ipadic-neologd-0.0.6/ \
    --word2vec_file ./data/jawiki.all_vectors.200d.txt \
    --batch_size 128 \
    --max_epoch 1 \
    --patience 5 \
    --max_len 512 \
    --optimizer Adafactor \
    --lr 0.0003 \
    --hidden_dropout 0.1 \
    --attention_dropout 0.3 \
    --use_pretrain_model \
    --use_cuda \
    --use_device_ids 0123

python ./util/ndcg.py ./result/$METHOD/$TYPE\_split/$SPLIT_ID/$TYPE/ ./data/chat_and_rec/$TYPE/
python ./util/rank_crr.py ./result/$METHOD/$TYPE\_split/$SPLIT_ID/$TYPE/ ./data/chat_and_rec/$TYPE/