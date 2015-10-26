
# coding: utf-8

# In[ ]:

# This code fits a model to the training HTML data set, and makes predictions on the test data set.
# The features are counts of various string occurences in each HTML file. 
#
# Counts of various HTML style indicators are included, inspired by David Schinn's public script.
# Counts of words chosen by FeatureSelection_Words.ipynb are also included.
#
import glob
import multiprocessing
import os
import re
import sys
import time
import numpy
import pandas as pd
#
from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
#
#
# The list feature_words contains words chosen by FeatureSelection_Words.
feature_words = ['home', 'com', 'end', 'tag', 'contact', 'blog', 'news', 'new','sign', 'content',
     'code', 'search', 'email', 'top', 'share', 'twitter','gt', 'site', 'video',
     'comments', 'script', 'div', 'footer', 'like', 'free', 'policy', 'one',
     'us', 'post', 'business', 'world', 'get', 'time', 'page', 'social', 'start',
     'posts', 'help', 'use', 'follow', 'health', 'class', 'best', 
     'life', 'header', 'ad', 'm', 'e', 'mobile', 'go', 'services']
#
#
# Function which takes input HTML file and counts occurences of various strings.
# create_data returns a dictionary values, where key = string, value = count.
#
def create_data(filepath):
    values = {}
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as infile:
        text = infile.read()
    # The following features are named to avoid potential overlap with feature_words    
    values['file_name'] = filename
    if filename in train_keys:
        values['sponsored0'] = train_keys[filename]
    else:
        values['sponsored0'] = 0 
    # Count occurences of various strings relating to HTML style
    values['lines0'] = text.count('\n')
    values['spaces0'] = text.count(' ')
    values['tabs0'] = text.count('\t')
    values['braces0'] = text.count('{')
    values['brackets0'] = text.count('[')
    values['words0'] = len(re.split('\s+', text))
    values['length0'] = len(text)
    values['hyperlinks0'] = text.count('<a')
    values['paragraphs0'] = text.count('<p')
    values['divs0'] = text.count('<div')
    values['urls0'] = text.count('http:') + text.count('https:') 
    values['images0'] = text.count('<img')
    values['@signs'] = text.count('@')
    values['hrefs'] = text.count('href')
    values['strong_'] = text.count('<strong')
    values['meta_'] = text.count('<meta')
    values['dotcom_'] = text.count('.com')
    values['link_'] = text.count('<link')
    values['script_'] = text.count('<script')
    values['function'] = text.count('function')
    # Count strings relating to social media
    values['facebook_'] = text.count('facebook') + text.count('Facebook')
    values['pinterest_'] = text.count('pinterest') + text.count('Pinterest')
    values['twitter_'] = text.count('twitter') + text.count('Twitter')
    values['instagram_'] = text.count('instagram') + text.count('Instagram')
    # Count occurences of numeric strings of the form " # ", where # ranges from 0 to 25
    number_string = [" " + str(i) + " " for i in range(26)]
    values['num_count'] = sum([text.count(item) for item in number_string])
    # Count occurences of single letters. Some HTML files have broken text, e.g. "Hello" -> "H e l l o"
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet_string = [" " + item + " " for item in alphabet]
    values['letter_count'] = sum([text.count(item) for item in alphabet_string])
    # Count occurences of words selected by FeatureSelection_Words (contained in feature_words)
    for word in feature_words:
        values[word] = text.count(word)
    return values
#
#
# Function which computes cross-validation score.
# Here we use the leave-P-out method, where P = (1-frac) x size of training data
# The model averages predictions from random forest and extremely randomized trees
#
# Note that the final submission uses n_estimators = 250
def cross_val(training_df,frac):
#    
    train_cv, test_cv = shuffle_and_sample(training_df,frac)            
#    
    rf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs = -1)
    rf.fit(train_cv[features], train_cv["sponsored0"])
    rf_pred = rf.predict_proba(test_cv[features])[:,1]
    del rf
#   
    et = ExtraTreesClassifier(n_estimators=100, random_state=1, n_jobs = -1)
    et.fit(train_cv[features], train_cv["sponsored0"])
    et_pred = et.predict_proba(test_cv[features])[:,1]
    del et
#
    test_probs = (rf_pred + et_pred)/2
    true_labels = test_cv["sponsored0"].values
    aucscore=roc_auc_score(true_labels,test_probs)
    return aucscore

# Function for sampling training data, used by cross_val
def shuffle_and_sample(df,frac):
    df = df.reindex(numpy.random.permutation(df.index))
    split = int(df.shape[0] * frac)
    train = df[:split]
    test = df[split:]
    return train, test


# In[ ]:

# This code block reads in training HTML data, and counts occurences of strings specified in create_data.
# The output is written to TrainingData.csv for posterity.
#
# Specify input directory containing test data
input_dir_train = "/media/sf_VboxShar/Native_Advertising_train/"
#
# Read in labels (sponsored = 1, non-sponsored = 0) for training data
train_labels = pd.read_csv("/media/sf_VboxShar/train_v2.csv")
train_keys = dict([a[1] for a in train_labels.iterrows()])
#
#
filepaths = glob.glob(input_dir_train + '*.txt')
num_tasks = len(filepaths)
#
p = multiprocessing.Pool()
results = p.imap(create_data, filepaths)
while (True):
    completed = results._index 
    print("\r--- Reading training HTML files: {:,} out of {:,}".format(completed, num_tasks), end='')
    sys.stdout.flush()
    time.sleep(1)
    if (completed == num_tasks): break
p.close()
p.join()
#
# Write output to TrainingData.csv
training_df = pd.DataFrame(list(results))
training_df.to_csv("TrainingData.csv", index=False)
del training_df
del results


# In[ ]:

# This code block reads in test HTML data, and counts occurences of strings specified in create_data.
# The output is written to TestData.csv for posterity.
#
# Specify input directory containing test data
input_dir_test = "/media/sf_VboxShar/Native_Advertising_test/"
#
filepaths = glob.glob(input_dir_test + '*.txt')
num_tasks = len(filepaths)
#
p = multiprocessing.Pool()
results = p.imap(create_data, filepaths)
while (True):
    completed = results._index 
    print("\r--- Reading test HTML files {:,} out of {:,}".format(completed, num_tasks), end='')
    sys.stdout.flush()
    time.sleep(1)
    if (completed == num_tasks): break
p.close()
p.join()
#
# Write output to TestData.csv
test_df = pd.DataFrame(list(results))
test_df.to_csv("TestData.csv", index=False)
del test_df
del results


# In[ ]:

# Load training and test data from csv files
training_df= pd.read_csv("TrainingData.csv")
test_df= pd.read_csv("TestData.csv")
# Obtain feature list
features = list(training_df.columns.values)
features.remove("sponsored0")
features.remove("file_name")


# In[ ]:

# Obtain cross-validation score, using leave-p-out where p = number of training files x 0.2
# 
aucscores=[cross_val(training_df,0.8) for i in range(10)]
mean_aucscore = sum(aucscores)/float(len(aucscores))
print("AUC score from cross validation: ", mean_aucscore)


# In[ ]:

# Make prediction for final submission, save to NativeAdv.csv
# 
rf = RandomForestClassifier(n_estimators=250, random_state=1, n_jobs = -1)
rf.fit(training_df[features], training_df["sponsored0"])
rf_pred = rf.predict_proba(test_df[features])[:,1]
del rf
#   
et = ExtraTreesClassifier(n_estimators=250, random_state=1, n_jobs = -1)
et.fit(training_df[features], training_df["sponsored0"])
et_pred = et.predict_proba(test_df[features])[:,1]
del et
#
predicted = 0.5 * rf_pred + 0.5 * et_pred
submission = pd.DataFrame({"file": test_df["file_name"], "sponsored": predicted})
submission.to_csv("NativeAdv.csv", index=False) 


# In[ ]:



