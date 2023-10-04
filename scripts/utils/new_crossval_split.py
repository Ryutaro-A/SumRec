from sklearn.model_selection import KFold,train_test_split
import random
import glob
import json

BASE_PATH = "./data/chat_and_rec/"
targets = ["all_topic", "only_trip", "no_trip"]

random.seed = 3
train_list = []
dev_list = []
test_list = []

for i in range(3):
    all_flist = []
    tmp_targets = targets[0:i]+targets[i+1:]
    print(tmp_targets)
    for t in tmp_targets:
        for fname in glob.glob(BASE_PATH + t + "/*.json"):
            tmp_path = fname.split("/")
            all_flist.append(tmp_path[-2]+'/'+tmp_path[-1])

    flist = []
    for fname in glob.glob(BASE_PATH + targets[i] + "/*.json"):
        tmp_path = fname.split("/")
        flist.append(tmp_path[-2]+'/'+tmp_path[-1])
    random.shuffle(flist)
    # print(len(flist))
    for n in range(5):
        split_data = []
        file_num = len(flist)
        file_batch = int(len(flist) / 10)+1
        # print(file_batch)
        kf = KFold(n_splits=10,shuffle=False)
        for train_index, test_index in kf.split(flist):
            # print(len(train_index))
            train_index,valid_index = train_test_split(train_index,test_size=file_batch)
            # print(len(train_index),len(valid_index),len(test_index))
            train_list = [flist[i] for i in train_index] + all_flist
            valid_list = [flist[i] for i in valid_index]
            test_list = [flist[i] for i in test_index]
            print(len(train_list), len(valid_list), len(test_list))
            # print(train_file)
            split_data.append({"train":train_list, "dev": valid_list, "test":test_list})
            # break

    with open("./data/crossval_split/re/" + targets[i] + "_split_" + str(n) + ".json", "w") as w:
        json.dump(split_data, w, ensure_ascii=False, indent=4)