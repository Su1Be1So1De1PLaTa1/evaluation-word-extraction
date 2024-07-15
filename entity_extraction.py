import spacy
import csv
import re
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np

data = []
dic = {}
count = 0
title = ["マギ","ゴールデンカムイ","進撃の巨人"]
magi = ["煌帝国","アル・サーメン","シンドリア","バルバッド","霧の団","マグノシュタット","レーム帝国","マギ","聖教連","抵抗軍","アリババ一味"]
golden = ["杉元一味","アイヌ","中央","第七師団","鶴見一派","土方一味","網走監獄","ロシア","極東ゲリラ"]
taitan = ["駐屯兵団"	,"調査兵団","エルディア復権派","訓練兵団","憲兵団","マーレ","エルディア人","イェーガー派","アズマビト","パラディ島","マーレ戦士","反マーレ派","タイバー家"]

number_list = []
txt_list = []
count = 0
for_count = 0
all_data = 339522

new_tokenizer = AutoTokenizer\
    .from_pretrained(f"C:/Users/lunad/Desktop/院研究用ファイル/sample-text-classification-bert")

new_model = (AutoModelForSequenceClassification
    .from_pretrained(f"C:/Users/lunad/Desktop/院研究用ファイル/sample-text-classification-bert"))

def hyouka_sys(text : str):
    new_tokenizer = AutoTokenizer\
    .from_pretrained(f"C:/Users/lunad/Desktop/院研究用ファイル/sample-text-classification-bert")

    new_model = (AutoModelForSequenceClassification
        .from_pretrained(f"C:/Users/lunad/Desktop/院研究用ファイル/sample-text-classification-bert"))
    value = 0
    inputs = new_tokenizer(text, return_tensors="pt")

    new_model.eval()

    with torch.no_grad():
        outputs = new_model(
            inputs["input_ids"], 
            inputs["attention_mask"],
        )

    y_preds = np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=1)
    def id2label(x):
        return new_model.config.id2label[x]
    y_dash = [id2label(x) for x in y_preds]
    

    if str(y_dash[0]) in "positive":
        value = 1
    elif str(y_dash[0]) in "negative":
        value = -1
    elif str(y_dash[0]) in "neutral":
        value = 2

    #値確認表
    #print(value)
    return value

def count_hyouka(title,name,count):
    global for_count
    done = 0
    for hihyouka in name:
        for hyouka in name:
            with open(rf"C:\Users\lunad\Desktop\研究検討メモ\data\組織分割{title}\Org_{hihyouka}.csv", 'r') as f:
                data = csv.reader(f)
                for i in data:
                    
                    if hyouka in i[7]:
                        result = hyouka_sys(i[10])
                        if int(number_list[count]) == result:
                            print(i[10])
                            txt_list.append(title+":"+i[10])
                            
                    for_count += 1
                    print(str(for_count)+":"+str(all_data))
            count += 1
def empty_del(str):
    if str == "":
        return False
    else:
        return True
    
for i in title:
    with open(rf"C:\Users\lunad\Desktop\院研究用ファイル\データ\隣接行列\評価結果 - {i}隣接行列.csv") as f:
        data = csv.reader(f)
        for number in data:
            del number[0]
            for k in number:
                number_list.append(k)


number_list = list(filter(empty_del,number_list))

for i in title:
    if i == "マギ":
        count_hyouka(i,magi,count)
        print("Task of magi is all done")
    elif i =="ゴールデンカムイ":
        count_hyouka(i,golden,count)
        print("Task of golden is all done")
    elif i == "進撃の巨人":
        count_hyouka(i,taitan,count)
        print("Task of Shingeki is all done")
    else:
        pass

        
#書き込み
f = open(f'C:/Users/lunad/Desktop/院研究用ファイル/結果/txt_by_evaluation.csv',"w")
data = txt_list
writer = csv.writer(f)
writer.writerow(data)
f.close()
    





