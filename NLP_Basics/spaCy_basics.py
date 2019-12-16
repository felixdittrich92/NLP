import spacy

nlp_de = spacy.load('de_core_news_sm')
#nlp_en = spacy.load('en_core_web_sm')

doc = nlp_de(u'BMW kauft U.S. AI-Startup für 6€ Millionen')

for token in doc:
    print(token.text, token.pos, token.pos_, token.dep_) # Verb/Substantiv/...

# Pipeline ausgeben
print(nlp_de.pipeline)
print(nlp_de.pipe_names)

doc2 = nlp_de(u'Gestern war´s   kalt in Deutschland')

for token in doc2:
    print(token.text, token.pos, token.pos_, token.dep_) # Verb/Substantiv/...

# Ausgabe für einzelne Elemente
print(doc2[0])
print(doc2[0].pos_)
print(type(doc[0]))