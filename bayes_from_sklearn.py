# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:13:53 2019

@author: Lenovo
"""

from sklearn.naive_bayes import MultinomialNB
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', max_features=500)
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
term_docs_train = cv.fit_transform(X_train)
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(term_docs_train, Y_train)
term_docs_test = cv.transform(X_test)
prediction = clf.predict(term_docs_test)
accuracy = clf.score(term_docs_test, Y_test)
print('The accuracy using MultinomialNB is: {0:.1f}%'.format(accuracy*100)) 