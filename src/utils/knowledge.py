from SPARQLWrapper import SPARQLWrapper
import json
from tqdm import tqdm as std_tqdm
from functools import partial
import time
import re
import itertools

tqdm = partial(std_tqdm, dynamic_ncols=True)

ng_list = [
    ':', '一覧', 'おける', '大会', '協会', '協議会', '学会',
]


def get_entity(word):
    try:
        sparql = SPARQLWrapper(endpoint='http://ja.dbpedia.org/sparql', returnFormat='json')
        sparql.setQuery(f"""
            PREFIX dbpedia-owl:  <http://dbpedia.org/ontology/>
            select distinct * where {{ <http://ja.dbpedia.org/resource/{word}> ?p ?o . }}
        """)
        results = sparql.query().convert()
        time.sleep(2)
    except:
        return get_entity(word)

    return results

def check(word):
    if re.search(r".月.日", word):
        return True
    if re.search(r".+?月", word):
        return True
    if re.search(r".+?年", word):
        return True
    if re.search(r".+?世", word):
        return True
    if re.search(r".+?語", word):
        return True
    if word.isdigit():
        return True
    if len(word) == len(word.encode('utf-8')):
        return True
    if re.fullmatch(r'[あ-ん]+?', word):
        return True
    if re.search(r'[あ-ん]行', word):
        return True
    if re.search('[!"#$%&\'().*+-/,:;<=>?@^_`{|}~]', word):
        return True
    if re.search('[・～、（）＝第]', word):
        return True
    if re.search(r'[0-9]', word):
        return True
    if word.count('の') >= 2:
        return True
    for ng_word in ng_list:
        if ng_word in word:
            return True

    return False

with open('./data/dbpedia/all_word.txt', encoding='utf-8') as f:
    all_word_list = f.read().rstrip('\n').split('\n')



# hop1_results = {
#     word: [result for result in get_entity(word)['results']['bindings']] \
#     for word in tqdm(all_word_list) if not check(word)
# }

# with open('./data/dbpedia/knowledge_data/hop-1.json', encoding='utf-8', mode='w') as f:
#     json.dump(hop1_results, f, indent=4, ensure_ascii=False)

with open('./data/dbpedia/knowledge_data/hop-1.json', encoding='utf-8') as f:
    hop1_results = json.load(f)


hop1_result_list = set([
    data['o']["value"].split('/')[-1].replace("Category:", "") \
    for key in hop1_results.keys() \
    for data in hop1_results[key] \
    if "wikiPageWikiLink" in data['p']["value"] and (not check(data['o']["value"].split('/')[-1].replace("Category:", "")))
])
print("hop1 length: ", len(hop1_result_list))



hop2_word_list = list(hop1_result_list - set(all_word_list))
print("hop2 length: ", len(hop2_word_list))

file_num = 0
interval = 10000
for i in range(0, len(hop2_word_list), interval):

    hop2_results = {
        word: [result for result in get_entity(word)['results']['bindings']] \
        for word in tqdm(hop2_word_list[i:i+interval])
    }

    with open(f'./data/dbpedia/knowledge_data/hop-2_{file_num}.json', encoding='utf-8', mode='w') as f:
        json.dump(hop2_results, f, indent=4, ensure_ascii=False)

    file_num += 1
