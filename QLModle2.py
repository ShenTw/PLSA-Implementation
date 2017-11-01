#!/usr/bin/env python

"""The simplest TF-IDF library imaginable.

Add your documents as two-element lists `[docname,
[list_of_words_in_the_document]]` with `addDocument(docname, list_of_words)`.
Get a list of all the `[docname, similarity_score]` pairs relative to a
document by calling `similarities([list_of_words])`.

See the README for a usage example.

"""
import math
import numpy as np
from random import random


def normalize(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0
    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s
        

class Document():
    def __init__(self,list_of_words):
        self.words = list_of_words

class Corpus():
    def __init__(self):
        self.documents=[]

    def add_document(self,document):
        self.documents.append(document)
    def build_vocabulary(self):
        discrete_set = set()
        for document in self.documents:
            for word in document.words:
                discrete_set.add(word)
            self.vocabulary = list(discrete_set)
                       
class QLM:

    def __init__(self):
        self.weighted = False
        self.documents = {}
        self.corpus_dict = {}
        self.trainedCorpus= {}
        self.sims = {}
        self.n_d = 18461
        self.n_t = 3
        self.n_w = 51253
        self.a = 0.9
        self.max_iter = 5 
    def initialTrain(self, corpus):
            # bag of words
            self.n_w_d = np.zeros([self.n_d, self.n_w],dtype = np.int)
            for di, doc in enumerate(corpus.documents):
                n_w_di = np.zeros([self.n_w],dtype = np.int)
                for word in doc.words:
                    if word in corpus.vocabulary:
                        word_index = corpus.vocabulary.index(word)
                        n_w_di[word_index]= n_w_di[word_index]+1
                self.n_w_d[di]= n_w_di
                
                
            # P(z|w,d)
            self.p_z_dw = np.zeros([self.n_d, self.n_w, self.n_t], dtype = np.float)
    		# P(z|d)
            self.p_z_d = np.random.random(size = [self.n_d, self.n_t])
            for di in range(self.n_d):
                normalize(self.p_z_d[di])
    		# P(w|z)
            self.p_w_z = np.random.random(size = [self.n_t, self.n_w])
            for zi in range(self.n_t):
                normalize(self.p_w_z[zi])
            #number_of_topics

    
                    
                    
                    
    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
        # 計算完每個字的次數
        length = float(len(list_of_words))
        
        # 處理頻率(除上該文章字數)
        for k in doc_dict:
            doc_dict[k] =math.log(doc_dict[k] / length)
          #其中一種作法  doc_dict[k] = 1 + doc_dict[k] / length
           # QLM中似乎用不到整個字典的字
        self.documents[doc_name]= doc_dict
        
    def train(self):
        print ("Training...")
        for i_iter in range(self.max_iter):

			# likelihood
        #    self.L = self.log_likelihood()



            print("Iter " + str(i_iter) )
            print("E-Step...")
            for di in range(self.n_d):
                for wi in range(self.n_w):
                    sum_zk = np.zeros([self.n_t], dtype = float)
                    for zi in range(self.n_t):
                        sum_zk[zi] = self.p_z_d[di, zi] * self.p_w_z[zi, wi]
                    sum1 = np.sum(sum_zk)
                    if sum1 == 0:
                        sum1 = 1
                    for zi in range(self.n_t):
                        self.p_z_dw[di, wi, zi] = sum_zk[zi] / sum1
            print ("M-Step...")
			# update P(z|d)
            for di in self.trainedCorpus:
                for zi in range(self.n_t):
                    sum1 = 0.0
                    sum2 = 0.0
                    for wi in range(self.n_w):
                        for word in range(self.n_t):
                            sum1 = sum1 + self.n_w_d[di,wi] * self.p_z_dw[di, wi, zi]
                            sum2 = sum2 + self.n_w_d[di,wi]
                    if sum2 == 0:
                        sum2 = 1
                    self.p_z_d[di, zi] = sum1 / sum2

			# update P(w|z)
            for zi in range(self.n_t):
                sum2 = np.zeros([self.n_w], dtype = np.float)
                for wi in range(self.n_w):
                    for di in range(self.n_d):
                        sum2[wi] = sum2[wi] + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
                sum1 = np.sum(sum2)
                if sum1 == 0:
                    sum1 = 1
                for wi in range(self.n_w):
                    self.p_w_z[zi, wi] = sum2[wi] / sum1
    
        print("train over")
        
    def likelihood(self, list_of_words, queryName):
        """Returns a list of all the [docname, similarity_score] pairs relative to a
list of words.

        """

        # building the query dictionary
        
        query_dict = {}
        for w in list_of_words:
            if query_dict.get(w,0)==0:
                query_dict[w] = query_dict.get(w, 0.0) + 1.0
        """ test for query_dict
        for w in query_dict:
            print(w)
        """
        # computing the list of similarities
        QLMDic = {}
        for doc in self.documents:
            #每一篇文件要做的事情
            #1. 讀出名字跟dic
            #2. 與query比對是否存在該字,有的話計算存下字典中，該字出現的機率值並相乘 沒有則不做 ,存在暫存值中
            #3. 把Likelihood中的值相乘所得到最後的值，並配合Doc名放到QLMDic中
            #4. 對QLMDic做排序    
            
            queryLikelihood= 0.0
            #1 得到該document的字典 {}
            dicTemp = self.documents[doc]
            
            #2.
            for w in query_dict:
                if w in dicTemp:
                        queryLikelihood = self.a*dicTemp[w] + (1-self.a)*float(self.corpus_dict[str(int(w))])+queryLikelihood

                    
            #3. with normalization
            QLMDic[doc] = queryLikelihood/float(len(list_of_words))
        
        #4.    
        self.sims[queryName] = sorted(QLMDic.items(), key=lambda d:d[1], reverse = True)
        #做排序    
         
            