
# coding: utf-8

# In[ ]:

# This code performs natural langauge processing on raw HTML files from the training data 
# The inputs are a random sample of some user-specified fraction of the ~337000 files in the training set.
#
#
# The output is given in WordCounts.csv, saved in the working directory.
# WordCounts.csv is then taken as an input for FeatureSelection-Words.ipynb
#
#
# WordCounts.csv has the following format:
#
# word_1 word_2 ... word_n filename
#   0      1           0   3093804_raw_html.txt
#   1      0           1   845185_raw_html.txt
#   ...   ...         ...     ...
#
# The boolean variable denotes whether or not that particular word appeared twice or 
# more in the visible text for a particular file. 
# The word list: word_1, word_2, ... word_n corresponds to common words, specified below in selected_words
#
# This cell contains function definitions and parameter initializations
# Input and output directories are sepecified in the next cell.
#
import re
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import csv
import glob
import multiprocessing
import sys
import time
import enchant
import random
from __future__ import print_function

# Initialize stopwords for natural language processing (taken from NLTK list)
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

# Use U.S. dictionary from enchanct
english_dict = enchant.Dict("en_US")


# Function which flattens a list
def flatten(list_):
    return [item for sublist in list_ for item in sublist]


# Function which returns True if "element" in HTML file is visible 
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True


# Function returns True if inputString satisfies selection criteria. 
#
# If selected_words is not empty, will only return true if inputString is in selected_words
#
# Else if selected_words is empty, will return true if inputString:
# (contains no numbers) && (not in stopwords) && (in english_dict or is a single character).
#
def Keep_String(inputString):
    if selected_words:
        selected = inputString in selected_words
    else:
        ContainsNumbers = bool(re.search(r'\d', inputString))
        stopword = inputString in stopwords
        single_char = len(inputString) == 1
        in_dictionary = english_dict.check(inputString)
        selected = not ContainsNumbers and not stopword and (single_char or in_dictionary)
#    
    return selected

# Function which counts words in word_list, and returns word count in dict format.
# Adds word counts to input dictionary in_dict (can be empty), and returns resulting dictionary.
# If unique = True, only apply word counting to unique word list.
#
def count_words(word_list,in_dict,unique):
    if unique:
        input_list = np.unique(word_list)
    else:
        input_list = word_list
    for word in input_list:
        if word in in_dict:
            in_dict[word] += 1
        else:
            in_dict[word] = 1
    return in_dict


# Function which takes in HTML file (filename) and tokenizes visible words.
# Output is list of resulting words, filtered with the Keep_String function
#
def get_text(filename):
    in_file = open(filename,'r')
    parser = BeautifulSoup(in_file, 'html.parser')
    texts = parser.findAll(text=True)
    visible_texts = filter(visible,texts)
    text_dump = []
# Lower and remove newlines
    for item in visible_texts:
        text_dump.append(str(item.encode('utf-8').lower()).strip('\n'))
# Remove punctiaton and split into words
    tokenized_words = filter(None,[re.sub("[^\w]", " ",item).split() for item in text_dump])
    word_list =flatten(tokenized_words)
    return filter(Keep_String,word_list)
#
#
# "Main" function. Takes in a HTML file (path specificed in variable filepath).
# Outputs dictionary to append to WordCounts.csv
#
def get_word_counts(filepath):
    # Obtain tokenized word list
    word_list = get_text(filepath)
    # Obtain output dictionary where: 
    # key = word in selected word, value = 1 if word appears at least twice, else value = 0
    output_dict = {}
    for word in selected_words:
        if word_list.count(word) > 1:
            output_dict[word] = 1
        else:
            output_dict[word] = 0
    # Add filename to dictionary
    output_dict["file_name"] = os.path.basename(filepath)
    return output_dict
#
#
selected_words = {}


# In[ ]:

# Set input directory (paths containing training data)
#
input_dir_train = "/media/sf_VboxShar/Native_Advertising_train/"


# In[ ]:

# Generate list of common words to keep, store in selected_words. 
#
# Obtain list by sampling 5000 files from training data.
#
WordSelectSamples = 5000
filepaths = glob.glob(input_dir_train + '*.txt')
random.shuffle(filepaths)
filepaths = filepaths[0:WordSelectSamples]
#
#
# Aim for ~1000 of the most common words, to keep WordCounts.csv from becoming huge.
#
#
# cum_dict is dictionary where: 
# key = word, value = number of files where word appeared at least twice
#
cum_dict = dict()
for i, filename in enumerate(filepaths):
#   get_text tokenizes visible words from raw the HTML file into a list, see function def. for details
    word_list = get_text(filename)
    cum_dict = count_words(word_list,cum_dict,unique = True)
    print("\r Reading file {:,} out of {:,}".format(i+1,WordSelectSamples), end='')
    sys.stdout.flush()
#    
# Convert cum_dict to dataframe
cum_count = pd.DataFrame(list(cum_dict.iteritems()),columns=["word_","word_count"])
#
# keep words that appear at least twice in more than mincount documents.
mincount=int(WordSelectSamples*0.065)
pruned_cum_count = cum_count[ ( cum_count["word_count"] > mincount)]
# 
selected_words = pruned_cum_count["word_"].tolist()
del cum_count
del cum_dict
del pruned_cum_count
print("  Number of selected words: {}".format(len(selected_words)))


# In[ ]:

from math import ceil
# Set fraction of training data to sample for WordCounts.csv
SampleFrac = 0.3
#
# Initialize paths for input files
filepaths = glob.glob(input_dir_train + '*.txt')
random.shuffle(filepaths)
NSamples = float(len(filepaths))*SampleFrac
NSamples = int(NSamples)
filepaths = filepaths[0:NSamples]
#
num_files = len(filepaths)
#
# Delete pre-existing WordCounts.csv
try:
    os.remove("WordCounts.csv")
except OSError:
    pass
#
#
batch_size = 2000
num_batches = ceil(float(num_files)/batch_size)
num_batches = int(num_batches)
# Begin reading input HTML files
for i in range(num_batches):
    p = multiprocessing.Pool()
    if i != (num_batches-1):
        filepaths_batch = filepaths[i*batch_size:(i+1)*batch_size]
    else:
        filepaths_batch = filepaths[i*batch_size:]
    results = p.imap(get_word_counts, filepaths_batch)
    while (True):
        completed = results._index
        progress = completed + i*batch_size
        print("\r--- Read {:,} out of {:,} HTML files".format(progress, num_files), end='')
        sys.stdout.flush()
        time.sleep(1)
        if (completed == batch_size): 
            break
        elif (progress == num_files):
            break
    p.close() 
    p.join()
    WordCounts = pd.DataFrame(list(results))
    if i==0:
        WordCounts.to_csv("WordCounts.csv", index=False, mode='a')
    else:
        WordCounts.to_csv("WordCounts.csv", index=False, mode='a', header=False)

