import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics

df = pd.read_csv('../TextFiles/moviereviews.tsv', sep='\t')
print(df.head())
print(len(df))
print(df['review'][0])
# auf 0 überprüfen
print(df.isnull().sum())
# 0er Einträge entfernen
df.dropna(inplace=True)
print(len(df))
# Spaces aus DataFrame entfernen
blanks = list()
for i, lb, rv in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)

print(blanks)
print(len(blanks))
df.drop(blanks, inplace=True)
print(len(df))

print(df['label'].value_counts()) 

# Modell trainieren
y = df['label']
X = df['review']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)
print(predictions)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))
