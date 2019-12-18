# POS - Part of Speech - Verb/Substantiv/Adjektiv/etc.
import spacy

nlp_de = spacy.load('de_core_news_sm')

doc = nlp_de(u'Der schnelle, braune Fuchs ist über den Rücken des faulen Hundes gesprungen.')
print(doc[12])
print(doc[12].pos_)
