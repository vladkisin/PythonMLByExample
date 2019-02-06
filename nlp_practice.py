# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:50:36 2019

@author: Lenovo
"""
#from nltk.corpus import names    #some of corpus import
#print(names.words()[:10])

#from nltk.stem.porter import PorterStemmer 
#porter_stemmer = PorterStemmer()           #stemming the tokens
#print(porter_stemmer.stem('learning'))

#from nltk.stem import WordNetLemmatizer   
#lemmatizer = WordNetLemmatizer()            #LEMMATIZING THE TOKENS
#print(lemmatizer.lemmatize('learning'))

#from sklearn.datasets import fetch_20newsgroups
#import numpy as np
#groups = fetch_20newsgroups()
#print(groups['target'])
#print(np.unique(groups.target))
#print(groups.data[0])
#print(groups.target[0], groups.target_names[groups.target[0]])
#import seaborn as sns
#sns.distplot(tuple(groups.target))

'''from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans

def letters_only(astr):
    return astr.isalpha()

cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in groups.data:
    cleaned.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
transformed = cv.fit_transform(cleaned)
km = KMeans(n_clusters=20)
km.fit(transformed)
labels = groups.target
plt.scatter(labels, km.labels_)
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.show()
'''
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF

def letters_only(astr):
    return astr.isalpha()

cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in groups.data:
    cleaned.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
transformed = cv.fit_transform(cleaned)
nmf = NMF(n_components=100, random_state = 43).fit(transformed)
for topic_idx, topic in enumerate(nmf.components_):
       label = '{}: '.format(topic_idx)
       print(label, " ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))