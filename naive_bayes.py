# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:41:38 2019

@author: Lenovo
"""
'''
file_path = 'enron1/ham/0007.1999-12-14.farmer.ham.txt'
with open(file_path, 'r') as infile:
    hamsample = infile.read()
    print(hamsample)
file_path2 = 'enron1/spam/0058.2003-12-21.GP.spam.txt'
with open(file_path2, 'r') as infile:
    hamsample = infile.read()
    print(hamsample)
'''
import os
import glob
file_path = 'enron1/spam/'
file_path_ham = 'enron1/ham/'
emails, labels = [], []
for filename in glob.glob(os.path.join(file_path_ham, '*.txt')):
    with open (filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open (filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

def letters_only(astr):
    return astr.isalpha()
def clean_data(emails):      
    from nltk.corpus import names
    from nltk.stem import WordNetLemmatizer
    cleaned = []
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    for email in emails:
        cleaned.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in email.split() if letters_only(word) and word not in all_names]))
    return cleaned
cleaned_emails = clean_data(emails)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', max_features=500)


def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index


def get_prior(label_index, total_count=0):
    prior = {label: len(index) for label, index in iter(label_index.items())}
    for i in prior.values():
        total_count += i
    for label in prior:
        prior[label] /= float(total_count)
    return prior

import numpy as np
def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, index in iter(label_index.items()):
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

def get_posterior(term_document_matrix, prior, likelihood, sum_posterior = 0):
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        for i in posterior.values():
            sum_posterior += i
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
            posteriors.append(posterior.copy())
    return posteriors[1::2]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
term_docs_train = cv.fit_transform(X_train)  
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train, label_index, smoothing=1)
term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_test, prior, likelihood)
correct = 0.0
for pred, actual in zip(posterior, Y_test):
    if actual == 1:
        if pred[1] >= pred[0]:
            correct += 1
    elif actual == 0:
        if pred[0] > pred[1]:
            correct += 1