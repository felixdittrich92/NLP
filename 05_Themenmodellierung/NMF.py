import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

reden = pd.read_csv('./TextFiles/politiker_reden.csv')

# deutsche Stoppwörter einlesen -> im Englischen in sklearn schon integriert
stop_words = pd.read_csv('./TextFiles/german_stopwords.txt', header=None)[0].values.tolist()

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
dtm = tfidf.fit_transform(reden['Artikel'])

nmf = NMF(n_components=6, random_state=42)
nmf.fit(dtm)

single_topic = nmf.components_[0]
top_word_indices = single_topic.argsort()[-10:]

for index in top_word_indices:
    print(tfidf.get_feature_names()[index])

# 6 Themen angegeben -> Ausgabe der dazugehörigen Wörter
for index, topic in enumerate(nmf.components_):
    print(f'Die TOP-15 Wörter für das Thema #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# Thema zum Artikel hinzufügen bzw. welches Thema hat die höchste Wahrscheinlichkeit im Artikel
topic_results = nmf.transform(dtm)
reden['Thema'] = topic_results.argmax(axis=1)
print(reden.head())