import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t') # tsv - tabseperated
print(df.head())
# Werte mit 0 ausgeben
print(df.isnull().sum())
# welche Labels
print(df['label'].unique())
# Aufteilung der Klassen
print(df['label'].value_counts())
# wichtigsten Metriken
print(df['length'].describe())

""" show
plt.xscale('log')
bins = 1,15**(np.arange(0,15))
plt.hist(df[df['label']=='ham']['punct'], bins = bins, alpha=0.8)
plt.hist(df[df['label']=='spam']['punct'], bins = bins, alpha=0.8)
plt.legend('ham', 'spam')
plt.show()
"""

X = df[['length', 'punct']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# Konfusionsmatrix / true/positive false/negative / true/negative false/positive
print(metrics.confusion_matrix(y_test, predictions))
# weitere Metriken
print(metrics.classification_report(y_test, predictions))
# Genauigkeit
print(metrics.accuracy_score(y_test, predictions))
