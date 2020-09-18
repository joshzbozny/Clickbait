import heapq
import json
import random
import numpy as np
import sklearn
import nltk
from nltk import bigrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import datasets
import text_classifier_kv as tc
import matplotlib.pyplot as plt
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
import Lemmatize
import joblib

lm = Lemmatize



ps = PorterStemmer()
targetData = [] #Array of 0 - clickbait - or 1 - nonclickbait, for each point in the dataset
weighting = []
totalclassif = {} #Total result from the classification data
classif = {} #Just the classification, clickbait or no-clickbait
rawData = {} #Array where the entire datapoint is stored. Includes title, article, author, everything
cleanedData = {} #Stemmed titles
totalDist = {} #Total distribution of the words in each title
filteredData = {} #Titles without stopwords
wordfreq = {} #Frequency of words in all titles combined
bigramfreq = {} #Frequency of bigrams in all titles combined
bigramTotalDist = {} #Total distribution of the bigrams in each title
lengths = {}
i = 0
stopwords = {}
# stopwords = {'to', 'the', ':', 'in', ',', 'of', "'s", 'a', 'for', "'", 'and', 'on', '’', 's', 'at', 'it', 'from', 'are',
#              '‘', 'you', 'as', 'be', 'thi', 'that', 'by', 'about', 'your', 'what', 'get', 'us', '.', 'he', "n't", 'not',
#              'ha', 'peopl', 'can', 'whi', 'wa', 'an'}
totalData = open('instances.jsonl')
classification = open('truth.jsonl')
# print(totalData)
for obj in totalData:  # For loop going through every object
    all_words = {}
    bigram_words = {}
    result = json.loads(obj)  # Makes the data readable, and without any unicode
    rawData[i] = result  # stores the result variable as a part of a huge array
    title = rawData[i].get('targetTitle') #The actual title
    stemmedTitle = ""
    cleanedTitle = ""
    tokenizedTitle = word_tokenize(title)  # Splits the title into its separate word components so it can be stemmed
    stemmedTitle = lm.lemmatize_sentence(title)
    # for w in tokenizedTitle: #Actually stems the title
    #     stemmedTitle = stemmedTitle + " " + ps.stem(w)
    stemmedTitle = stemmedTitle.lower()
    k = rawData[i].get('id')
    cleanedData[k] = stemmedTitle  # cleanedData[k] is the stemmed title with id k
    lengths[k] = len(cleanedData[k])

    for w in cleanedData[k].split(): #Removes the stopwords
        if w not in stopwords:
            cleanedTitle = cleanedTitle + " " + w
    filteredData[k] = cleanedTitle #Stores the titles without stopwords in the filteredData array
    for w in filteredData[k].split(): #Frequency count of all words in the dataset
        if w not in wordfreq.keys():
            wordfreq[w] = 1
        else:
            wordfreq[w] += 1
        if w not in all_words.keys():
            all_words[w] = 1
        else:
            all_words[w] += 1

    totalDist[k] = all_words

    for h in bigrams(word_tokenize(filteredData[k])): #Frequency count of all bigrams in the dataset
        if h not in bigramfreq.keys():
            bigramfreq[h] = 1
        else:
            bigramfreq[h] += 1
        if h not in bigram_words.keys():
            bigram_words[h] = 1
        else:
            bigram_words[h] += 1

    bigramTotalDist[k] = bigram_words
    i += 1
    if (i == 19538):  # This will end the processing after the list of titles in the dataset, which is approximately 2/3 of the data
        break


most_freq = heapq.nlargest(10000, wordfreq, key=wordfreq.get)
# print(most_freq)

bigram_most_freq = heapq.nlargest(2000, bigramfreq, key=bigramfreq.get)

freqWords = joblib.dump(most_freq, 'frequencies.sav')
# print(bigram_most_freq)
i = 0
for obj in classification: #Adds the classification, clickbait or no-clickbait, to each of the datapoints
    result = json.loads(obj)
    totalclassif[i] = result
    theClass = result.get('truthClass')
    # adjustedClass = result.get('truthMean')
    # if adjustedClass>=.39:
    #     theClass = 'clickbait'
    # else:
    #     theClass = 'no-clickbait'
    classif[result.get('id')] = theClass
    i += 1

sentence_vectors = []

for i in filteredData:
    sent_vec = []
    for token in most_freq:
        if token in totalDist[i]:
            sent_vec.append(totalDist[i].get(token))
        else:
            sent_vec.append(0)
    if(classif[i] == 'clickbait'):
        targetData.append(0)#('clickbait')
        weighting.append(3.09)
    else:
        targetData.append(1)#('noclickbait')
        weighting.append(1)
    sent_vec.append(lengths[i])
    sentence_vectors.append(sent_vec)

sentence_vectors = np.asarray(sentence_vectors)
targetData = np.asarray(targetData)
train = sentence_vectors[:13000]
test = sentence_vectors[13000:]
targetTrain = targetData[:13000]
targetTest = targetData[13000:]



#Attempt to do bigrams instead of unigrams

# bigram_vectors = []
# for i in filteredData:
#     bigram_vec = []
#     for token in bigram_most_freq:
#         if token in bigramTotalDist[i]:
#             bigram_vec.append(bigramTotalDist[i].get(token))
#         else:
#             bigram_vec.append(0)
#     bigram_vectors.append(bigram_vec)
#     if (classif[i] == 'clickbait'):
#         targetData.append(0) #clickbait
#     else:
#         targetData.append(1) #non-clickbait
#         # print(bigram_vec)
#
#
# bigram_vectors = np.asarray(bigram_vectors)
#
# train = bigram_vectors[:13000]
# test = bigram_vectors[13000:]
# targetTrain = targetData[:13000]
# targetTest = targetData[13000:]

sm = SMOTE('minority')
trainRefit, targetTrainRefit = sm.fit_resample(train, targetTrain)
testRefit, targetTestRefit = sm.fit_resample(test, targetTest)




#Multinomial Naive Bayes
multinom = MultinomialNB()
multifit = multinom.fit(trainRefit, targetTrainRefit)

exp = targetTrain
pred = multinom.predict(testRefit) #train or test
multiprob = multinom.predict_proba(testRefit) #train or test
scorey = multinom.score(testRefit, targetTestRefit) #train or test
prob = [item[0] for item in multiprob]

print(metrics.classification_report(targetTestRefit, pred, digits = 2)) #train or test
print(metrics.confusion_matrix(targetTestRefit, pred))
print(roc_auc_score(targetTestRefit, prob)) #train or test
print(multinom.get_params())
print(np.mean(pred == targetTestRefit))
print("---------------------------")


#GridSearch
grid = []
count = 0
for i in np.arange(.05, .95, .05):
    save = [i, 1-i]
    grid.append(save)


grid_params = {
   'alpha': np.linspace(0.001, 1, 100),
   'class_prior': grid,
   'fit_prior': [True, False]
}

search = RandomizedSearchCV(estimator = multinom, param_distributions= dict(grid_params), cv = 5, scoring = 'balanced_accuracy')# refit = 'precision_score')
#search = GridSearchCV(estimator = MultinomialNB(), param_grid= dict(grid_params), cv = 10, scoring = 'balanced_accuracy', n_jobs = -1)
search.fit(trainRefit, targetTrainRefit)
#print(search.best_params_)
gridPred = search.predict(testRefit)


gridProb = search.predict_proba(testRefit)[:,1] #0 is clickbait


print("GridSearch Classification")
print(metrics.classification_report(targetTestRefit, gridPred, digits = 2))
print(roc_auc_score(targetTestRefit, gridProb))
print("----------------------------")






# filename = 'finalized_model.sav'
# joblib.dump(search, filename)

font = {'family': 'serif',
        'name': 'Times New Roman',
        'size': 26
        }
titleFont ={'family': 'serif',
            'name': 'Times New Roman',
        'size': 32
        }
fpr, tpr, thresholds = roc_curve(targetTestRefit, gridProb)
print("False positive rate")
for i in fpr:
    print(i)
print("------------------------------")
print("True positive rate")
for i in tpr:
    print(i)

plt.plot(fpr, tpr, linestyle = '--', label = 'Grid')
plt.xlabel('False Positive Rate', fontdict = font)
plt.ylabel('True Positive Rate', fontdict = font)
plt.title('Receiver operating characteristic curve', fontdict = titleFont)
plt.show()




#Attempt to randomly shuffle data for the 2/3 1/3 split

#random.shuffle(rawData)
# rawData = pd.DataFrame(rawData)
# trainingData = rawData.iloc[:13000]
# testingData = rawData.iloc[13000:]