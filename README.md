# Kaggle_NativeAdvertising
Code used for the Kaggle competition "Truly native" sponsored by DATO. 
Finished 71/274. See https://www.kaggle.com/c/dato-native for competition details.

The goal of this competition is to develop an algorithim to distinguish 
native advertising from non-sponsored content, using ~30GB of training data.
The training data consists of raw HTML text files from web pages obtained 
through www.stumbleupon.com. 

This repo contains the following files:

1. HTMLScrape-NLP.ipynb- Performs natural language processing on training HTML files. 
   Outputs WordCounts.csv, used as input for FeatureSelection_Words.ipynb

2. FeatureSelection_Words.ipynb- Performs feature selection on English words in HTML files.
  Reads WordCounts.csv. outputs list of words (best_words) to be included as features for model.

3. NativeAd-Classifier.ipynb- Code used to generate submission. Prediction on test set made using average of 
  Random Forest and Extremely Randomized Trees. Takes counts of various string in HTML file as features.
  Counts of words in best_words from FeatureSelection_Words.ipynb are also included as features.

The Python codes corresponding to the iPython notebooks are also included.
