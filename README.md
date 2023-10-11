# SumRec

This is the official implementation of the following paper: Ryutaro Asahara, Masaki Takahashi, Chiho Iwahashi and Michimasa Inaba. SumRec: A Framework for Recommendation using Open-Domain Dialogue. PACLIC 2023.

Comming Soon.

>Abstract<br>
>Chat dialogues contain considerable useful information about a speaker’s interests, preferences, and experiences.
>Thus, knowledge from open-domain chat dialogue can be used to personalize various systems and offer recommendations for advanced information..This study proposed a novel framework SumRec for recommending information from open-domain chat dialogue.
>The study also examined the framework using ChatRec, a newly constructed dataset for training and evaluation \footnote{Our dataset and code is publicly available at https://github.com/Ryutaro-A/SumRec . 
>To extract the speaker and item characteristics, the SumRec framework employs a large language model (LLM) to generate a summary of the speaker information from a dialogue and to recommend information about an item according to the type of user.
>The speaker and item information are then input into a score estimation model, generating a recommendation score.
>Experimental results show that the SumRec framework provides better recommendations than the baseline method of using dialogues and item descriptions in their original form.

## Overview
The code and sample data for our work is organized as:

* `src/` contains the main model and evaluation scripts
* `data/chat_and_rec/` has our dataset
* `data/prompts/` has our prompt we used to generate dialogue summary and recommendation information.
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

Runinng train, test, evaluations scripts if you excecute `train_sumrec.sh`.

```.bash
sh train_sumrec.sh
```

If you want to try baseline, you can excute `train_baseline.sh`.
```.bash
sh train_baseline.sh
```

## License
This software is released under the MIT License, see LICENSE.txt.

## Contacts

Twitter: [@Ryu_pro_m](https://twitter.com/Ryu_pro_m)

Mail: ryu1104.as[at]gmail.com
