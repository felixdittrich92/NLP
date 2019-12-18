# POS - Part of Speech - Verb/Substantiv/Adjektiv/etc.
import spacy
from spacy import displacy

nlp_de = spacy.load('de_core_news_sm')

doc = nlp_de(u'Der schnelle, braune Fuchs ist über den Rücken des faulen Hundes gesprungen.')
print(doc[12])
print(doc[12].pos_)

# Ausgabe aller Wortarten aus Text + Erklärung
for token in doc:
    print(f'{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_):{10}}:')

# Wortarten zählen
POS_count = doc.count_by(spacy.attrs.POS)
print(POS_count)
print(doc.vocab[84].text) # hier sind im Text z.B.: 3 Adjektive vorhanden

# alle ausgeben
for k, v in sorted(POS_count.items()):
    print(f'{k}. {doc.vocab[k].text}: {v}')

print('------------------------------------------')

# für einzelne Wörter je nach Textinhalt/Sinn
doc = nlp_de(u'Floh er über die Brücke?')
token = doc[0]
print(f'{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_):{10}}:')


doc = nlp_de(u'Sprang der Floh über die Brücke?')
token = doc[2]
print(f'{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_):{10}}:')

# Visualisierung
doc = nlp_de(u'Der schnelle, braune Fuchs ist über den Rücken des faulen Hundes gesprungen.')
displacy.serve(doc, style='dep', options={'distance':100, 'color': 'red', 'bg':'#09a3d5', 'font':'Times'})

doc2 = nlp_de(u'Dies ist ein Satz. Dies ist ein anderer Satz. Ein dritter Satz.')
spans = list(doc2.sents)
displacy.serve(spans, style='dep', options={'distance':100, 'color': 'red', 'bg':'#09a3d5', 'font':'Times'})