export CUDA_VISIBLE_DEVICES=1
python src/trainers.py \
    --pretrained_model_name nlp-waseda/roberta-large-japanese-seq512-with-auto-jumanpp \
    --use_pretrain_model \
    --plm_hidden_dropout 0.0 \
    --plm_attention_dropout 0.0 \
    --data_type all_topic \
    --split_info_dir ./data/crossval_split_5/ \
    --data_dir ./data/ \
    --unseen \
    --dialogue_max_length 512 \
    --desc_max_length 512 \
    --lr 1e-6 \
    --warmup_steps 500 \
    --epochs 10 \
    --train_batch_size 32 \
    --optimizer adafactor \
    --test_batch_size 32 \
    --use_cuda --cuda_num 1 \
    --split_id 3 --max_try 5 --use_double_encoder --use_summary