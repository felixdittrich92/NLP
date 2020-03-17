import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
a = 'This was a good movie'
score = sid.polarity_scores(a)

print(score)

a = 'This was the best, most awesome movie EVER MADE!!!'
score = sid.polarity_scores(a)

print(score)

a = 'This was the worst film to ever disgrace the screen'
score = sid.polarity_scores(a)

print(score)

# Anhand von echten Daten
df = pd.read_csv('./TextFiles/amazonreviews.tsv', sep='\t')
print(df.head())

print(df['label'].value_counts())
df.dropna(inplace=True)

blanks = []

for i,lb,rv in df.itertuples(): # Iteriere über den DataFrame
    if type(rv)==str:           # Vermeide NaN-Werte
        if rv.isspace():        # Teste 'review' auf Leerzeichen
            blanks.append(i)    # Füge der Liste passende Indizes hinzu

df.drop(blanks, inplace=True)

print(df['label'].value_counts())
a = df.loc[0]['review']  # erste Zeile aus dem DataFrame
score = sid.polarity_scores(a)

print(score)

# Scores für das ganze DataFrame berechnen und hinzufügen
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
print(df.head())

# Value von compound holen
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
print(df.head())

# compound score berechnen (Anzeige ob positiv oder negativ)
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg')
print(df.head())

# Genauigkeit ausgeben
print(accuracy_score(df['label'], df['comp_score']))
# Übersicht ausgeben
print(classification_report(df['label'], df['comp_score']))
# Konfusionsmatrix ausgeben
print(confusion_matrix(df['label'], df['comp_score']))