# SumRec

This is the official implementation of the following paper: Ryutaro Asahara, Masaki Takahashi, Chiho Iwahashi and Michimasa Inaba. SumRec: A Framework for Recommendation using Open-Domain Dialogue. PACLIC 2023.

>Abstract<br>
>Chat dialogues contain considerable useful information about a speaker’s interests, preferences, and experiences.
>Thus, knowledge from open-domain chat dialogue can be used to personalize various systems and offer recommendations for advanced information..This study proposed a novel framework SumRec for recommending information from open-domain chat dialogue.
>The study also examined the framework using ChatRec, a newly constructed dataset for training and evaluation \footnote{Our dataset and code is publicly available at https://github.com/Ryutaro-A/SumRec . 
>To extract the speaker and item characteristics, the SumRec framework employs a large language model (LLM) to generate a summary of the speaker information from a dialogue and to recommend information about an item according to the type of user.
>The speaker and item information are then input into a score estimation model, generating a recommendation score.
>Experimental results show that the SumRec framework provides better recommendations than the baseline method of using dialogues and item descriptions in their original form.

## Overview
The code and sample data for our work is organized as:

* `scripts/` contains the main model and evaluation scripts
* `data/chat_and_rec/` has our dataset
* `data/crossval_split/` has a json file specifying the division method for cross-validation


## Requirements
1. The implementation is based on Python 3.x. To install the dependencies used, run:
```.bash
pip install -r requirements.txt
```
2. Install Juman++ 2.0.0-rc3
```.bash
sudo apt update -q
sudo apt install -qy cmake g++ make wget xz-utils

wget "https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz"
tar xvf jumanpp-2.0.0-rc3.tar.xz

cd jumanpp-2.0.0-rc3
mkdir bld && cd bld

curl -LO https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp
mv catch.hpp ../libs/
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install -j "$(nproc)"
sudo make install
```

## Get Started

Runinng train, test, evaluations scripts if you excecute `run.sh`.

```.bash
sh run.sh
```

## Options

When changing the method or data type, change the options according to the table below.

| Args               | Desctiption                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| data_type          | You can chosse no_restriction or travel or except_for_travel.                                                          |
| method             | You can chosse human or tfidf_cossim or w2v_cossim, or w2v_svr or bert-base or roberta-base or roberta-large.         |
| data_dir           | Please fill in the data directory.                                                                                       |
| split_info_dir     | Enter the location of the directory containing the file that describes how the data was split.                               |
| split_id           | Enter the division ID of the data.                                                                                           |
| model_output_dir   | Enter the location where you would like to save the model.                                                                               |
| output_dir         | Enter the location where you want to store the results of the predictions made by the chosen method.                                                                       |
| vocab_dir          | Enter the directory of the dictionary to be used for the pre-trained Transformer.(`./data/roberta_dic/` or `./data/BERT/`)                                                        |
| mecab_dict         | Enter the specified directory for mecab dictionaries used for tfidf and word2vec.                                               |
| word2vec_file      | Enter the path to your pre-studied word2vec.                                                                             |
| batch_size         | Batch size when training a Transformer model.                                                                            |
| max_epoch          | Epoch size when training a Transformer model.                                                                             |
| patience           | Epoch size until learning is terminated if the minimum loss is not updated.                                           |
| max_len            | Maximum token size including dialog history and tourist attraction descriptions.                                                         |
| optimizer          | Optimization function used to train the Transformer model.                                                            |
| lr                 | Transformer model learning rate.                                                                                |
| hidden_dropout     | Dropout rates for the hidden layer of the Transformer model.                                                              |
| attention_dropout  | Dropout rate for the attention layer of the Transformer model.                                                         |
| use_pretrain_model | With this option, the GPU is used to train the Transformer model. Without this option, only the model structure is used.           |
| use_cuda           | With this option, the GPU is used to train the Transformer model.                                                   |
| use_device_ids     | By specifying this option and a number, you can specify the ID of the GPU device to be used for training. If not specified, all GPUs are used. |

## License
This software is released under the MIT License, see LICENSE.txt.

## Contacts

Twitter: [@Ryu_pro_m](https://twitter.com/Ryu_pro_m)

Mail: ryu1104.as[at]gmail.com
