# Lemmatisierung bezieht sich beim Wortstamm finden auf den 
# Satzzusammenhang -> funktioniert besser als Stemming

import spacy

nlp_de = spacy.load('de_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

doc1 = nlp_de(u'Ich bin ein Wettläufer und renne in einem Wettlauf, weil ich es liebe zu rennen seitdem ich heute gerannt bin.')
doc2 = nlp_en(u'I saw teen mice today !')

#for token in doc1:
    #print(token.text, '\t', token.pos_, '\t', token.lemma_, '\t')


def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')

show_lemmas(doc1)
print('\t')
show_lemmas(doc2)

