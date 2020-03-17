import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

questions = pd.read_csv('./TextFiles/quora_questions.csv')
print(questions.head())

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(questions['Question'])

nmf = NMF(n_components=20, random_state=42)
nmf.fit(dtm)

for index, topic in enumerate(nmf.components_):
    print(f'Die TOP-15 Wörter für das Thema #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

topic_results = nmf.transform(dtm)
questions['Topic'] = topic_results.argmax(axis=1)
print(questions.head())