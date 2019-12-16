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

# Spacy Erklärung - GENIAL :)
print(spacy.explain('mo'))

# weitere Ausgaben
print(doc2[1].text) # normale Wort
print(doc2[1].lemma_) # auf welches das Wort zurückzuführen ist
print(doc2[1].tag_) # Wortart/-verbindung
print(spacy.explain('VAFIN'))
print(doc2[1].shape_) # Wortlänge in Form von x´en X-Großbuchstabe x-Kleinbuchstabe
print(doc2[1].is_alpha) # True wenn alphanummerisch
print(doc2[1].is_stop) # ob Stoppwort ist

doc3 = nlp_de(u'Der nach John Lennon benannte Flughafen von Liverpool\
zeigt in der Eingangshalle mehrere großflächige Zitate von Lennon-Songs,\
darunter Imagine. Auch das zentrale Flughafen-Motto "Above us only sky" ist ein Zitat.')

life_quote = doc3[24:30]
print(life_quote)
print(type(life_quote))

doc4 = nlp_de(u'Dies ist der erste Satz. Dies ist ein neuer Satz. Dies ist der letzte Satz.')

for sentence in doc4:
    print(sentence)

print(doc[4].is_sent_start)