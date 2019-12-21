# Textkörper in Array transformieren
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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

