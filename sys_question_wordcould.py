# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:25:31 2018

@author: TobeyWang
畫關建字雲
"""

#取資料庫
import sqlite3
import pandas
import jieba.analyse
#畫圖
from wordcloud import WordCloud
import matplotlib.pyplot as plt
jieba.set_dictionary("dict.txt.big")
value=""
#自建立專有名詞詞庫
query1='SELECT 詞庫,重要性,詞性 FROM MI_EXCEL_KeyWords '
with sqlite3.connect('\\Datasource\\nlp_sys.sqlite') as db:
    df_keywords=pandas.read_sql(query1,con=db)
df_keywords.to_csv(r'sys_userdict.txt', header=None, index=None, sep=' ', mode='w+',encoding='utf8') 
jieba.load_userdict("sys_userdict.txt")
##自建立專有名詞詞庫
#jieba.load_userdict("userdict.txt")
#topK 前幾大關鍵字
#分析百大問題
query1='SELECT 提問 FROM MI_EXCEL_QUESTION_IMPORT '
with sqlite3.connect('\\Datasource\\nlp_sys.sqlite') as db:
    df_questions=pandas.read_sql(query1,con=db)
df_questions.to_csv(r'sys_source.txt', header=None, index=None, sep=' ', mode='w+',encoding='utf8') 
content = open('sys_source.txt', 'rb').read()
tags = jieba.analyse.extract_tags(content,topK=20,withWeight=True)
for tag, weight in tags:
   if value=="":
       value=tag
   else :
       value=value+" "+tag
   a=str(weight)
   print(tag , "," ,str(int(weight *10000)))
##取出關鍵字 
##topK代表要取的關鍵字次數(tf-idf)
#with open("source.txt", "rb") as f:
#    for line in f:
#        tags = jieba.analyse.extract_tags(line,topK=10,withWeight=True)
#        for tag, weight in tags:
#            print(tag + "," + str(int(weight *10000)))

#取得斷詞結果
with open("sys_keyword.txt", "w+", encoding="utf-8") as f:
    sl=jieba.cut(content)
    value2word="/".join(sl)
    f.write(value2word)
print("取得斷詞結果",value)
#設定文字雲 
stopword_set = set()
with open('stopword.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
#stop_word_content = " ".join(stopword_set)
wc = WordCloud(font_path="NotoSansCJKtc-Black.otf", 
#設置字體
background_color="white", #背景顏色
max_words = 2000 , #文字雲顯示最大詞數
stopwords=stopword_set) #停用字詞
wc.generate(value)
#詞雲轉為圖片存檔
wc.to_file("sys_wordcloud.jpg")
#顯示詞雲
plt.imshow(wc)
plt.axis("off")
plt.figure(figsize=(10,6), dpi = 100)
plt.show()
