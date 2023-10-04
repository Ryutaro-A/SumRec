from sklearn.model_selection import KFold
import glob
import json
import random
import os
import sys

args = sys.argv

# 話者の偏りを少なくしてtrain dev testに分割
BASE_PATH = "data/chat_and_rec/"

targets = ["all_topic", "no_trip", "only_trip"]

# random.seed = 3
train_list = []
dev_list = []
test_list = []

# dev, testに含む最大の同じ話者のデータ
max_same_speaker_count = 4

# 同じく都道府県ファイル名
max_same_pref_count = 2

test_sid_count = {}
test_pref_count = {}
dev_sid_count = {}
dev_pref_count = {}
    
for t in targets:
    
    flist = []
    for fname in glob.glob(BASE_PATH + t + "/*.json"):
        #print(fname.rsplit("/")[-1])
        #print(os.path.basename(fname))
        #print(fname)
        #flist.append(fname.rsplit("/")[-1])
        fname_extraxted_path = "/".join(fname.rsplit("/")[1:])
        flist.append(fname_extraxted_path)
    print(len(flist))
    
    file_num = len(flist)
    
    # test
    tmp_test = []
    
    for _ in range(1000):
        tmp = random.choice(flist)
        tmp_path = "/".join(tmp.rsplit("/")[:-1])
        tmp = tmp.rsplit("/")[-1]
        id1 = tmp.split("_")[0]
        id2 = tmp.split("_")[1]
        pref = tmp.split("_")[2]
        
        if id1 not in test_sid_count:
            test_sid_count[id1] = 0
        if id2 not in test_sid_count:
            test_sid_count[id2] = 0
        if pref not in test_pref_count:
            test_pref_count[pref] = 0
        
        if test_sid_count[id1] >= max_same_speaker_count or test_sid_count[id2] >= max_same_speaker_count or test_pref_count[pref] > max_same_pref_count:
            continue
        else:
            test_sid_count[id1] += 1
            test_sid_count[id2] += 1
            test_pref_count[pref] += 1
            tmp = tmp_path+"/"+tmp
            flist.remove(tmp)
            tmp_test.append(tmp)
            
            addcount = 1 if t == "all_topic" else 0
            if len(tmp_test) >= int(file_num  * 0.1) + addcount:
                print(len(tmp_test), "test ok")
                test_list += tmp_test
                break
    
    # dev
    tmp_dev = []
    for _ in range(1000):
        tmp = random.choice(flist)
        tmp_path = "/".join(tmp.rsplit("/")[:-1])
        tmp = tmp.rsplit("/")[-1]
        id1 = tmp.split("_")[0]
        id2 = tmp.split("_")[1]
        pref = tmp.split("_")[2]
        
        if id1 not in dev_sid_count:
            dev_sid_count[id1] = 0
        if id2 not in dev_sid_count:
            dev_sid_count[id2] = 0
        if pref not in dev_pref_count:
            dev_pref_count[pref] = 0
        
        if dev_sid_count[id1] >= max_same_speaker_count or dev_sid_count[id2] >= max_same_speaker_count or dev_pref_count[pref] > max_same_pref_count:
            continue
        else:
            dev_sid_count[id1] += 1
            dev_sid_count[id2] += 1
            dev_pref_count[pref] += 1
            tmp = tmp_path+"/"+tmp
            flist.remove(tmp)
            tmp_dev.append(tmp)
            
        addcount = 1 if t == "all_topic" else 0
        if len(tmp_dev) >= int(file_num * 0.1) + addcount:
            print(len(tmp_dev), "dev ok")
            dev_list += tmp_dev
            break
    train_list += flist
    
    
print(len(train_list), len(dev_list), len(test_list))
split_data = {"train":train_list, "dev": dev_list, "test":test_list}

k = args[2]
with open(f"data/crossval_split/{targets[0]}_split_{k}.json", "w") as w:
    json.dump(split_data, w, ensure_ascii=False, indent=4)


    # for n in range(5):
    #     split_data = []
    #     random.shuffle(flist)
    #     for test_split in range(split_num):
            
    #         dev_split = test_split + 1
    #         if dev_split >= split_num:
    #             dev_split = 0

            
    #         for i in range(split_num):
    #             b = int(round(batchsize*i, 0))
    #             e = int(round(batchsize*(i+1), 0))
    #             split = flist[b:e]
    #             if i == test_split:
    #                 test_list += split
    #             elif i == dev_split:
    #                 dev_list += split
    #             else:
    #                 train_list += split
    #         split_data.append({"train":train_list, "dev": dev_list, "test":test_list})
    #         print(len(train_list), len(dev_list), len(test_list))

# with open(BASE_PATH + "crossval_split/" + t + "_split_" + str(n) + ".json", "w") as w:
#     json.dump(split_data, w, ensure_ascii=False, indent=4)

