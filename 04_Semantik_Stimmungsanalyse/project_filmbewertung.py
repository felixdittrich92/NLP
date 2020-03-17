import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('./TextFiles/moviereviews.tsv', sep='\t')
print(df.head())

df.dropna(inplace=True)

blanks = []

for i,lb,rv in df.itertuples(): # Iteriere über den DataFrame
    if type(rv)==str:           # Vermeide NaN-Werte
        if rv.isspace():        # Teste 'review' auf Leerzeichen
            blanks.append(i)    # Füge der Liste passende Indizes hinzu

df.drop(blanks, inplace=True)

print(df['label'].value_counts())

sid = SentimentIntensityAnalyzer()
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg')

print(accuracy_score(df['label'], df['comp_score']))
print(classification_report(df['label'], df['comp_score']))
print(confusion_matrix(df['label'], df['comp_score']))