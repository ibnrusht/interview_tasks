import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer


# stemming
def stemming(string: str):
    stop_words = ['в', 'с', 'и', 'о', 'к', 'у', 'за', 'по', 'из', 'для',
                  'от', 'не', 'из-за', 'на', 'до', 'об', 'во']
    stemmer = SnowballStemmer("russian")
    text = re.sub('[^A-Za-z0-9 а-яА-я+:]+', '', string.lower())
    text = [item for item in text.split() if item not in stop_words]
    text = [stemmer.stem(y) for y in text]
    text = ' '.join([i for i in text])
    return text


df = pd.read_csv("dataset/train.tsv", sep="\t")

# remove stop-words
banned = ['в', 'с', 'и', 'о', 'к', 'у', 'за', 'по', 'из', 'для',
          'от', 'не', 'из-за', 'на', 'до', 'об', 'во']
f = lambda x: ' '.join([item for item in x.lower().split() if item not in banned])
df['title'] = df['title'].apply(f)

# creating train and test datasets
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# training classifier
clf = Pipeline([('cnv', CountVectorizer(preprocessor=stemming, ngram_range=(1,2))),
      ('tfidf', TfidfTransformer(use_idf=False)),
      ('nb', MultinomialNB(alpha=0.4))])
clf.fit(X_train['title'], X_train['is_fake'])
predicted = clf.predict(X_test['title'])

print(classification_report(X_test['is_fake'], predicted))
