import csv
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

stop_words = stopwords.words('english')

x=pd.read_csv('articles.csv')

cols=x.columns

regex='[^A-Za-z]'

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values=line.split()
    word=values[0]
    embedding_for_word = np.asarray(values[1:], dtype='float64')
    word_embeddings[word]=embedding_for_word
    
f.close()

for row in x.itertuples():
    
    claps=row[2]
    article=row[6]
    title=row[5]

    sentences=sent_tokenize(article)[:-2]
    new_sentences=[re.sub(regex,' ',i) for i in sentences] 
    clean_sentences = [s.lower() for s in new_sentences]
    
    # remove stop words.
    # cleaned_article=[]
    # for sen in clean_sentences:
    #     words=sen.split()
    #     new_s=[]
    #     for w in words:
    #         if w not in stop_words:
    #            new_s.append(w)
    #     new_without_stop=' '.join(new_s)
    #     cleaned_article.append(new_without_stop)

    # vectorise each sentence.
    embedded_sentences=[]
    for sent in clean_sentences:
        words=sent.split()
        num_words=len(words)
        sum_array=np.zeros((100,),dtype='float64')
        for w in words:
            
            try:
                embedding_of_w=word_embeddings[w]
                sum_array+=embedding_of_w
            except KeyError:
                embedding_of_w=np.random.randn(100)
                sum_array+=embedding_of_w
                # print(embedding_of_w.size)

        embed_of_sentence=sum_array/(num_words+0.001) 
        embedded_sentences.append(embed_of_sentence)
    
    # prepare cosine similarity matrix.
    sim_mat = np.zeros([len(embedded_sentences), len(embedded_sentences)])
    for i in range(len(embedded_sentences)):
        for j in range(len(embedded_sentences)):
            if i!=j:
                sim_mat[i][j]=cosine_similarity(embedded_sentences[i].reshape(1,100),embedded_sentences[j].reshape(1,100))[0,0] 

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    # ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(embedded_sentences)))

    sorted_scores=set()
    for i,s in enumerate(clean_sentences):
        s_c=scores[i]
        sorted_scores.add((s_c,s))
        
    m=sorted(sorted_scores,reverse=True)  
    for i in range(4):
        print(m[i][1],'\n')  
        
    # print(sorted_scores[:10])
            
    
    break                   