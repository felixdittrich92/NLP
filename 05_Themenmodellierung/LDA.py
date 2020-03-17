import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

reden = pd.read_csv('./TextFiles/politiker_reden.csv')
print(reden.head())

# deutsche Stoppwörter einlesen -> im Englischen in sklearn schon integriert
stop_words = pd.read_csv('./TextFiles/german_stopwords.txt', header=None)[0].values.tolist()
# max_df 95% der Dokumente verwenden min_df mind. 2 Dokumente verwenden  
# stop_words im Englischen : stop_words='english'

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)

dtm = cv.fit_transform(reden['Artikel'])
lda = LatentDirichletAllocation(n_components=6, random_state=42)
lda.fit(dtm)
# Liste aller vorhandenen Wörter
#print(cv.get_feature_names())

# Top 20 Wörter aus den Dokumenten
single_topic = lda.components_[0]
top_words_indices = single_topic.argsort()[-20:]
print(top_words_indices)

for index in top_words_indices:
    print(cv.get_feature_names()[index])

# 6 Themen angegeben -> Ausgabe der dazugehörigen Wörter
for index, topic in enumerate(lda.components_):
    print(f'Die TOP-15 Wörter für das Thema #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# Thema zum Artikel hinzufügen bzw. welches Thema hat die höchste Wahrscheinlichkeit im Artikel
topic_results = lda.transform(dtm)
reden['Thema'] = topic_results.argmax(axis=1)
print(reden.head())