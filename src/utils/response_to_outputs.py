import glob
import json
import os
import time


data_type_list = [
    # 'all_topic',
    # 'only_trip',
    'no_trip'
]
data_dir = './outputs/result/chatgpt_pre/dialogue/'
row_dir = './data/GPT35sum_rec_chat_and_rec/'
out_dir = "./outputs/result/chatgpt/dialogue/"


for data_type in data_type_list:
    files = glob.glob(data_dir + data_type + '/*.txt')

    for file_path in files:
        print(file_path)
        with open(file_path, encoding='utf-8') as f:
            lines = f.read().split('\n\n')

        with open(row_dir + data_type + '/' + os.path.basename(file_path).replace(".txt", ".json"), encoding='utf-8') as f:
            json_data = json.load(f)

        speakers = list(json_data["questionnaire"].keys())
        result_dic = {
            speakers[0]: [],
            speakers[1]: []
        }
        for i, line in enumerate(lines):
            fix_line = line.replace("Bさん:", "").replace("\n", "")
            print(fix_line)
            print()
            result_dic[speakers[0]].append({
                "id": i+1,
                "score": float(fix_line[0])
            })

            result_dic[speakers[1]].append({
                "id": i+1,
                "score": float(fix_line[1])
            })



        os.makedirs(out_dir + data_type, exist_ok=True)
        with open(out_dir + data_type + '/' + os.path.basename(file_path).replace(".txt", ".rmd.json"), mode='w', encoding='utf-8') as f:
            json.dump(result_dic, f, indent=4, ensure_ascii=False)
