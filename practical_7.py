import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text for processing
text = "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."

# 1. Sentence Tokenization
tokenized_text = sent_tokenize(text)
print("Sentence Tokenization:\n", tokenized_text)

# 2. Word Tokenization
tokenized_word = word_tokenize(text)
print("Word Tokenization:\n", tokenized_word)

# 3. Removing Punctuations and Stop Words
stop_words = set(stopwords.words("english"))
text_cleaned = re.sub('[^a-zA-Z]', ' ', text)
tokens = word_tokenize(text_cleaned.lower())
filtered_text = [w for w in tokens if w not in stop_words]
print("Filtered Text:\n", filtered_text)

# 4. Stemming
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered_text]
print("Stemmed Words:\n", stemmed_words)

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_text]
print("Lemmatized Words:\n", lemmatized_words)

# 6. POS Tagging
pos_tags = nltk.pos_tag(filtered_text)
print("POS Tagging:\n", pos_tags)

# 7. TF-IDF Calculation
documentA = "Jupiter is the largest Planet"
documentB = "Mars is the fourth planet from the Sun"
bagOfWordsA = documentA.split()
bagOfWordsB = documentB.split()
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
numOfWordsA = dict.fromkeys(uniqueWords, 0)
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
for word in bagOfWordsB:
    numOfWordsB[word] += 1

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)

def computeIDF(documents):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

idfs = computeIDF([numOfWordsA, numOfWordsB])

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

df = pd.DataFrame([tfidfA, tfidfB])
print("TF-IDF DataFrame:\n", df)
