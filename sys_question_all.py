# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:46:32 2018

@author: TobeyWang
"""

# -*- coding: utf8 -*-
"""
Created on Mon Oct  8 11:30:47 2018

@author: TobeyWang

 gensim 建立詞庫
"""
import jieba.analyse
#字符編碼 package codecs >ASCII的字符串
#import codecs
#先將jieba的詞庫建立成字典
jieba.set_dictionary("dict.txt.big")
#自建立專有名詞詞庫
jieba.load_userdict("sys_userdict.txt")
#動態建立
#jieba.add_word(word,freq=None,tag=None)
wf=open("sys_lyrics_cut.dataset", mode="w",encoding="utf8")
#取得斷詞結果
with open("sys_source.txt", "r",encoding="utf8") as f:
    for line in f:
        words = jieba.cut(line)
        wf.write(" ".join(words))
        #print(" ".join(words))
wf.close()
#同義字(這裡忽略)
#word_net = []
#with open("lyrics/word_net.txt", "r") as f1:
#    for line in f1:
#        word_net.append(line)
#
#word_net = sorted(set(word_net))
#word_net_dic = {}
#
#for word in word_net:
#    word_s = word.split()
#    word_net_dic[word_s[0]] = word_s[1]

# 將資料處理後的檔案存檔
#lyrics_cut_cathaysite.dataset :jieba斷詞結果
#lyrics_word_net_cathaysite.dataset :同義字取代之結果
#lyrics_cathaysite.dict：最後的字典 dictionary 語料庫
#lyrics_cathaysite.mm：corpus 序列化語料庫
#lyrics_cathaysite.lsi list 索引庫
wf = open("sys_lyrics_word_net.dataset", "w",encoding="utf8")

with open("sys_lyrics_cut.dataset", "r",encoding="utf8") as f2:
    for line in f2:
        line_words = line.split()
        line_lyrics = ""
        for line_word in line_words:
            #同義字取代
#            if line_word in word_net_dic:
#                line_lyrics = line_lyrics + word_net_dic[line_word] + ' '
#            else:
                line_lyrics = line_lyrics + line_word + ' '
        #print(line_lyrics+"\n")
        wf.write(line_lyrics+"\n")

wf.close()
#看詞典結果
with open("sys_lyrics_cut.dataset",encoding="utf8") as fn:
    for line in fn:
        print(line)
#建模        
import logging
log_filename = "createworddic.log"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,filename=log_filename)
import os
from gensim import corpora, models, similarities
#停用詞庫
#stopword_set = {}.fromkeys(["舊戶","新戶","一個","什麼","那個"]) #停用詞，要被忽略的詞
stopword_set = set()
with open('stopword.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

stop_word_content = " ".join(stopword_set)
dictionary = corpora.Dictionary(document.split() for document in open("sys_lyrics_word_net.dataset",encoding="utf8"))

stoplist = set(stop_word_content.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id] #dictionary.token2id: 代表什麼字詞對應到什麼id，有幾個id就代表有幾維向量空間
dictionary.filter_tokens(stop_ids) # 移除停用字
dictionary.compactify() #remove faps in id sequence after worfs that were removed
dictionary.save("sys_lyrics_dic.dict")

###查看字典裡的資料
#for word,index in dictionary.token2id.items(): 
#    print(word +" id:"+ str(index))
#    break
##打開詞庫，
#texts = [[word for word in document.split() if word not in stoplist]
#         for document in open("lyrics_word_net_cathaysite.dataset",encoding="utf8")]
##將 corpus 序列化
#corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.serialize("lyrics_cathaysite.mm", corpus) # Corpus in Matrix Market format 


#tf-idf轉換與Lsi model
# 載入語料庫(save 辭庫會有亂碼的問題)
if (os.path.exists("sys_lyrics_dic.dict")):
    dictionary = corpora.Dictionary.load("sys_lyrics_dic.dict")
#    corpus = corpora.MmCorpus("lyrics_cathaysite.mm") # 將數據流的語料變為內容流的語料
    
    ##查看字典裡的資料
    for word,index in dictionary.token2id.items(): 
        print(word +" id:"+ str(index))
        break
    #打開詞庫，
    texts = [[word for word in document.split() if word not in stoplist]
             for document in open("sys_lyrics_word_net.dataset",encoding="utf8")]
    #將 corpus 序列化
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize("sys_lyrics_corpus.mm", corpus) # Corpus in Matrix Market format     
    print("載入語料庫完成")
else:
    print("語料庫不存在")
# 創建 tfidf model
tfidf = models.TfidfModel(corpus)
# 轉為向量表示
corpus_tfidf = tfidf[corpus]
# 創建 LSI model
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
#lsi = LsiModel.load('lyrics_cathaysite.lsi')  # load model
corpus_lsi = lsi[corpus_tfidf] # LSI潛在語義索引
lsi.save('sys_lyrics_model.lsi') #保存模型
corpora.MmCorpus.serialize('sys_lyrics_corpus.mm', corpus_lsi)
print("查看：每一筆理由在各主題的佔比計算:")
print(lsi.print_topics(5))
## 查看：每一筆理由在各主題的佔比計算
#for doc in corpus_lsi:
#    print(doc)
#相似度計算 
#隨便輸入一串字並經過斷詞
doc=input("what are your want to say?")
#doc="聲明書去哪下載"
words_new = jieba.cut(doc)
#words_new=" ".join(words_new)
vec_bow = dictionary.doc2bow(words_new) # 把doc語料庫轉為一個一個詞包(斷詞轉換成陣列)
print("把doc語料庫轉為一個一個詞包 結果：")
print(vec_bow)
vec_lsi = lsi[vec_bow] # 用前面建好的 lsi 模型去計算這一篇歌詞 (input: 斷詞後的詞包、output: 20個主題成分)
print("Lis計算結果：")    
print(vec_lsi)    
# 建立索引
index = similarities.MatrixSimilarity(lsi[corpus]) 
index.save("sys_lyrics_index.index") 

# 計算相似度（前五名）
sims = index[vec_lsi] 
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print("計算相似度前五名：")   #可設定前五相關度的部分先不理他 
print(sims[:5])    

lyrics = [];
fp = open("sys_lyrics_word_net.dataset",encoding="utf8") # 斷詞後的歌詞
#fp = open("lyrics/lyrics.dataset") # 看完整的歌詞
for i, line in enumerate(fp):
    lyrics.append(line)
fp.close()

res="";
for lyric in sims[:5]:
    flo=float(lyric[1])
    if(res=='' and flo>0.5):
        res=lyrics[lyric[0]]
    print("\n相似問題：",  lyrics[lyric[0]])
    print("相似度：",  lyric[1])

import re
import sqlite3
import pandas
likequestion=re.sub('\s','',res)
query="SELECT 回答 FROM MI_EXCEL_QUESTION_IMPORT where 提問='"+likequestion+"'"
with sqlite3.connect('\\Datasource\\nlp_sys.sqlite') as db:
    df_result=pandas.read_sql(query,con=db)
print('you say：',doc)
if df_result.size>=1:
    print('robot say：',df_result["回答"].iloc[0])
else:
   print('robot say：我不明白你的明白')     