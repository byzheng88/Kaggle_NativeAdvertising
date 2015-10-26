
# coding: utf-8

# In[ ]:

# This code takes WordCounts.csv from NativeAd-Classifier as input, and selects words to be used as features
#
# Feature selection is performed by fitting a Random Forest model to a subset of words in WordCounts.csv, 
# and then choosing words which have the largest importances as computed by the Random forest model.
#
#
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#
# Read in word counts, store list of words in wordlist
WordCounts = pd.read_csv("WordCounts.csv")
wordlist = list(WordCounts.columns.values)
wordlist.remove("file_name")
#
# Add a column "sponsored_" to WordCounts. Equals 1 if sponsored content, else equals 0.
# Labels for the training data are provided in train_v2.csv
#
train_labels = pd.read_csv("/media/sf_VboxShar/train_v2.csv")
train_keys = dict([a[1] for a in train_labels.iterrows()])
sponsored_list = []
for name in WordCounts["file_name"]:
    if name in train_keys:
        sponsored_list.append(train_keys[name])
    else:
        sponsored_list.append(0)    
WordCounts["sponsored_"] = pd.Series(sponsored_list)


# In[ ]:

# Construct data frame WordFreqs with columns word, freq
# freq is frequency of files where word appears more than once.
word_freq = []
num_files = float(len(WordCounts))
for word in wordlist:
    count = (WordCounts[word].sum())
    word_freq.append([word,float(count/num_files)])
WordFreqs = pd.DataFrame(word_freq,columns = ["word","freq"])
#
# Construct Dataframe including only the 100 most common words
most_common_words = list(WordFreqs.sort(columns = ["freq"], ascending = False).iloc[:100]["word"])
most_common_words_df = WordFreqs[WordFreqs["word"].isin(most_common_words)]
print "Most common words/letters:", most_common_words


# In[ ]:

# 
# Fit Random forest to WordCounts, where target variable is "sponsored_" 
# and features are listed in most_common_words
# 
print "Fitting random forest model"
rf = RandomForestClassifier(n_estimators=100, random_state = 1)
rf.fit(WordCounts[most_common_words],WordCounts["sponsored_"])
#
# Obtain feature importances for a given word, store in array word_scores
importances = rf.feature_importances_
word_scores = []
for i, item in enumerate(importances):
    word_scores.append([most_common_words[i],item]) 
# Output 45 words with the highest feature importance scores
# These words will be used as features in the final model in NativeAd-Classifier
#
#
word_scores.sort(key=lambda x: x[1],reverse=True)
best_words = [item[0] for item in word_scores[0:45]]
print "Chosen features:", best_words


# In[ ]:

# In another version of this code, a similar analysis was performed.
# However instead of taking the 150 most common words, I took the 150 words with the highest Logdiff,
# Logdiff = abs(log(sponsored_freq) - log(unsponsored_freq)).
#
# sponsored_freq is the fraction of sponsored HTML file which contain a given word more than once.
# unsponsored_freq is similarly defined.
#
# This analysis selects the additional features listed below. 
best_words_2 = ['ad','m', 'e', 'mobile', 'go', 'services']
# Adding these features to best_words improves both the cross-validation and leaderboard scores.
# NativeAd-Classifier takes the words listed in both best_words and best_words_2 as features.

