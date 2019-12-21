# Erkennung von bestimmten Wörtern in Texten z.B.: Namen, Firmen (Microsoft etc.) , Geld, Zeit und Datum
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy import displacy

nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
    else:
        print('Keine benamten Entitäten gefunden.')

# keine
doc = nlp_en(u'Hi how are you ?')
show_ents(doc)

# mit Entitäten
doc = nlp_en(u'May I go to Washington, DC next May to see the Washington Monument.')
show_ents(doc)

doc = nlp_en(u'Tesla to build a U.K. factoy for $6 million.')
show_ents(doc)

# weitere Entitäten manuell hinzufügen
ORG = doc.vocab.strings[u'ORG'] # Anzahl gelistete Organisations in nlp_en
print(ORG)

new_ent = Span(doc, 0, 1, label=ORG) # Wort Tesla aus doc / 0 starttoken(wort) 1 Endtoken (wort)
doc.ents = list(doc.ents) + [new_ent]

show_ents(doc)    # Tesla hinzugefügt  

# mehrere Entitäten auf einmal hinzufügen
doc = nlp_en(u'Our company plans to introduce a new vacuum cleaner.'
             u'If successful, the vacuum-cleaner will be our first product.')

show_ents(doc)

matcher = PhraseMatcher(nlp_en.vocab)
phrase_list = ['vacuum-cleaner', 'vacuum cleaner']
phrase_patterns = [nlp_en(text) for text in phrase_list]
matcher.add('new_product', None, *phrase_patterns) # *entpacken wichtig

matches = matcher(doc)
print(matches)

PROD = doc.vocab.strings[u'PRODUCT']
new_ents = [Span(doc, match[1], match[2], label=PROD) for match in matches]
doc.ents = list(doc.ents) + new_ents

show_ents(doc)

# zählen von Entitäten im Text
doc = nlp_en(u'Originally priced at $29.60, the sweater was marked down to five dollars.')
print(len([ent for ent in doc.ents if ent.label_ == 'MONEY']))


# Visualisierung
doc_de = nlp_de(u'Ich wohne in Berliner')
displacy.serve(doc_de, style='ent')

doc_en = nlp_en(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.'
                u'By contrast, Sony sold only 7 thousand Walkman music players.')

displacy.serve(doc_en, style='ent')

# Satzweise anzeigen
for sent in doc_en.sents:
    displacy.serve(nlp_en(sent.text), style='ent') #jupyter=True für Verwendung im Notebook

# Anpassungen
colors = {'ORG': 'linear-gradient(45deg, red, blue)'} # radial-gradient ohne deg angabe
options = {'ents': ['ORG', 'PRODUCT'], 'colors': colors}

displacy.serve(doc_en, style='ent', options=options)
