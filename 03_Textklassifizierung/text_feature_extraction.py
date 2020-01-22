# Textkörper in Array transformieren
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Wortschatz aufbauen Text 1
vocab = {}
i = 1
with open('../TextFiles/1.txt') as f:
    x = f.read().lower().split()

for word in x:
    if word in vocab:
        continue
    else:
        vocab[word] = i
        i+=1
print(vocab)

# Wortschatz aufbauen Text 2
with open('../TextFiles/2.txt') as f:
    x = f.read().lower().split()

for word in x:
    if word in vocab:
        continue
    else:
        vocab[word] = i
        i+=1
print(vocab)

# Vektor erzeugen mit Platz für jedes Wort
one = ['../TextFiles/1.txt']+[0]*len(vocab)
print(one)

two = ['../TextFiles/2.txt']+[0]*len(vocab)
print(two)

# speichern der Häufigkeit jedes Wortes aus 1.txt in Vektor
with open('../TextFiles/1.txt') as f:
    x = f.read().lower().split()
for word in x:
    one[vocab[word]]+=1
print(one)

with open('../TextFiles/2.txt') as f:
    x = f.read().lower().split()
for word in x:
    two[vocab[word]]+=1
print(two)

print('\n')
print('-------Jetzt mit ScikitLearn Implementierung-------')

df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
y = df['label']
X = df['message']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# (Zeilen, Wörter)
print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# beste , da Kombination 
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
print(X_train_tfidf.shape)

# Modell trainieren 
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

# Pipeline anlegen / alles in einem 
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)
print(predictions)
# Konfusionsmatrix
print(metrics.confusion_matrix(y_test, predictions))
# weitere Metriken
print(metrics.classification_report(y_test, predictions))
# Genauigkeit
print(metrics.accuracy_score(y_test, predictions))

test = text_clf.predict(['Congratulations! You have been selected as winner.TEXT WON to 44255'])
print(test)
