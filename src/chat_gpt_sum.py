import glob
import json
import openai
import os
import time


openai.api_key = ""

def open_json(file_path, split_id):
    with open(file_path, mode="r", encoding="utf-8") as f:
        return [os.path.basename(data).replace(".json", "").replace(".rmd", "") for data in json.load(f)[split_id]["test"]]


def generate(prompt):
    time.sleep(1)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {
                        "role": "user",
                        "content":prompt
                    },
                ],
                temperature=0.0,
            )
            break
        except:
            time.sleep(30)
            continue

    return response["choices"][0]["message"]["content"]


row_test_files = [
    # open_json("./crossval_split_5/all_topic_split.json", 3),
    open_json("./data/crossval_split_5/unseen/only_trip_split.json", 1),
    open_json("./data/crossval_split_5/unseen/no_trip_split.json", 1),
]

data_type_list = [
    # 'all_topic',
    'only_trip',
    'no_trip'
]
data_dir = './data/GPT35sum_chat_and_rec/'
out_dir = "./outputs/result/chatgpt_pre/sum/"

with open('./data/prompt/sum.txt', encoding='utf-8') as f:
    base_prompt = f.read()

out_files = [os.path.basename(data).replace(".rmd.json", ".json") for data in glob.glob(out_dir + '*/*.json')]

for data_type, row_test in zip(data_type_list, row_test_files):
    files = glob.glob(data_dir + data_type + '/*.json')
    print("files:", len(files))
    test_files = [file_path for file_path in files if os.path.basename(file_path).replace(".json", "").replace(".rmd", "") in row_test]
    print("test:", len(test_files))
    for file_path in test_files:
        filename = os.path.basename(file_path)
        
        # 既にあるファイルは飛ばす
        if filename in out_files:
            # print('すでにあります')
            continue
        print(filename)
        with open(file_path, encoding='utf-8') as f:
            json_data = json.load(f)

        result_dic = {}
        for speaker in list(json_data["questionnaire"].keys()):
            result_dic[speaker] = []
            for i, place_dict in enumerate(json_data["place"]):

                prompt = base_prompt \
                        + "\n\n\n--6--\n" \
                        + '【話者の特徴】\n' \
                        + json_data["summary"][speaker] +'\n\n' \
                        + '【観光地の説明】\n' \
                        + place_dict["description"] + '\n\n' \
                        + '【スコア】\n'

                sys_mes = generate(prompt)
                print(sys_mes)
                try:
                    score = float(sys_mes)
                except:
                    score = 3.0
                result_dic[speaker].append({
                    "id": str(i+1),
                    "score": score,
                })

        print(file_path)
        os.makedirs(out_dir + data_type, exist_ok=True)
        with open(out_dir + data_type + '/' + os.path.basename(file_path).replace(".json", ".rmd.json"), mode='w', encoding='utf-8') as f:
            json.dump(result_dic, f, indent=4, ensure_ascii=False)

        print()
