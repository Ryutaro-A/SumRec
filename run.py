import sys

sys.path.append("w2v_tfidf")
import argparse
import tfidf_cossim
import w2v_cossim
import w2v_svr
import human

sys.path.append("pretrained-transformer/roberta")
sys.path.append("pretrained-transformer/bert")
import roberta, bert

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", help="対話&評価のjsonファイルのディレクトリ")
parser.add_argument("--output_dir", help="推定結果を出力するディレクトリ")
parser.add_argument("--split_info_dir", help="分割方法のjsonファイル")
parser.add_argument("--split_id", help="分割ID", type=int)
parser.add_argument("--data_type", help="データの種類")
parser.add_argument("--model_output_dir", help="モデルを保存するディレクトリ")
parser.add_argument("--method", help="手法を指定． [uniform, random, tfidf_cossim, tfidf_svr, w2v_cossim, w2v_svr, w2v_wmd, human, oracle")

# word2vecを使用するモデル用
parser.add_argument("--word2vec_file", help="word2vecモデルのファイルパス")
parser.add_argument("--mecab_dict", help="neologdなどのmecabの追加辞書のディレクトリ")

# 事前学習済みTransformer用
parser.add_argument("--vocab_dir", help="tokenizerのvocabファイルのディレクトリ")
parser.add_argument("--use_pretrain_model", action='store_true')
parser.add_argument("--use_cuda", action='store_true')
parser.add_argument("--use_device_ids", type=str, default="all")
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--hidden_dropout", type=float, default=0.0)
parser.add_argument("--attention_dropout", type=float, default=0.0)
parser.add_argument("--optimizer", type=str, default=0.0)


args = parser.parse_args()

if not args.data_dir.endswith("/"):
    args.data_dir += "/"
if not args.output_dir.endswith("/"):
    args.output_dir += "/"
if not args.model_output_dir:
    args.model_output_dir += "/"

if args.method == "tfidf_cossim":
    tfidf_cossim.run(args)
elif args.method == "w2v_cossim":
    w2v_cossim.run(args)
elif args.method == "w2v_svr":
    w2v_svr.run(args)
elif args.method == "human":
    human.run(args)
elif args.method == "bert-base":
    bert.run(args)
elif args.method == "roberta-base":
    roberta.run(args, False)
elif args.method == "roberta-large":
    roberta.run(args, True)