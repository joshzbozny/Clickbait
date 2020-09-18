import joblib
import nltk
import Lemmatize
import json
import numpy as np
import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

lm = Lemmatize

loaded_model = joblib.load('finalized_model.sav')
commonWords = joblib.load('frequencies.sav')

def CleanSentence(pred):
    stemmedTitle = lm.lemmatize_sentence(pred)
    stemmedTitle = stemmedTitle.lower()
    return stemmedTitle


#NEED TO CHANGE ARRAY SHAPE TO 2D ARRAY.
def Vectorize(pred):
    sentence_vector = []
    wordfreq = {}
    title = CleanSentence(pred)
    for w in title.split():
        if w not in wordfreq.keys():
            wordfreq[w] = 1
        else:
            wordfreq[w] += 1
    for token in commonWords:
        if token in title.split():
            sentence_vector.append(wordfreq.get(token))
        else:
            sentence_vector.append(0)
    sentence_vector.append(len(title))
    sentence_vector = np.asarray(sentence_vector)
    return sentence_vector


def PredictClass(pred, model = loaded_model):
    vec = Vectorize(pred)
    predict = loaded_model.predict_proba([vec])
    predict = predict[0]
    # print(predict)
    if predict[0] <=.4:
        return False
    else:
        return True
# PredictClass("You won't believe what happened!")
# PredictClass("THESE 7 WEIRD TRICKS FOR CLICKBAIT TITLES WILL CHANGE YOUR LIFE")
# PredictClass("this is why dogs follow you into the bathroom. I never knew this!")
#
# PredictClass("Aggregating machine learning and rule based heuristics for named entity recognition")
# PredictClass("See how an Alaskan glacier has shrunk over time")
# PredictClass("A dance of two atoms reveals chemical bonds forming and breaking")
# PredictClass("Fluid dynamics may help drones capture a dolphin’s breath in midair")

# PredictClass("Propagation from Deceptive News Sources Who Shares, How Much, How Evenly, and How Quickly?")