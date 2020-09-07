import textract

import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy import displacy

nlp_de = spacy.load('de_core_news_lg')

text = textract.process('/home/felix/Desktop/Test/AVB_Kollektiv_SBB_Billet_National_International_D.pdf', encoding='ascii')
doc = str(text).strip()
doc = nlp_de(doc)

match = ["ORG"]

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                if ent.text.endswith(('AG', 'GmbH', 'GMBH', 'ag')):
                    print(ent.text + ' - ' + ent.label_)
            elif ent.label_ == 'DATE':
                print(ent.text + ' - ' + ent.label_)
    else:
        print('Keine benamten Entitäten gefunden.')

show_ents(doc)