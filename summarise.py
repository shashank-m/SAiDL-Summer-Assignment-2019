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

regex='[^A-Za-z]' # all characters that don't belong in A-Z or a-z.

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values=line.split()
    word=values[0]
    embedding_for_word = np.asarray(values[1:], dtype='float64')
    word_embeddings[word]=embedding_for_word
    
f.close()
for row in x.itertuples(): # each row contains data for a article.
    
    claps=row[2]
    article=row[6]
    title=row[5]

    sentences=sent_tokenize(article)[:-2] # last two sentences are generally common to all articles and not an integral part to the summary.
    new_sentences=[re.sub(regex,' ',i) for i in sentences] #replace special chars with blank space.
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
            except KeyError: # for out of vocabulary words.
                embedding_of_w=np.random.randn(100)
                sum_array+=embedding_of_w
        # average of all word embeddings in each sentence.
        embed_of_sentence=sum_array/(num_words+0.001) 
        embedded_sentences.append(embed_of_sentence)
    
    # prepare cosine similarity matrix. Finds the cos(theta) between vectors to find out how similar they are.
    sim_mat = np.zeros([len(embedded_sentences), len(embedded_sentences)])
    for i in range(len(embedded_sentences)):
        for j in range(len(embedded_sentences)):
            if i!=j:
                sim_mat[i][j]=cosine_similarity(embedded_sentences[i].reshape(1,100),embedded_sentences[j].reshape(1,100))[0,0] 

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph) # scores each sentence of the article based on pagerank algorithm.
    
    sorted_scores=set()
    for i,s in enumerate(clean_sentences):
        s_c=scores[i]
        sorted_scores.add((s_c,s))
        
    m=sorted(sorted_scores,reverse=True) # sort scores from highest to lowest.  
    for i in range(6): # 6 represents top 6 sentences.
        print(m[i][1],'\n')  
    print('#################################') # end of each article's summary.       
        
                
                     