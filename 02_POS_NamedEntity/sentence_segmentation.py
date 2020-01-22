import spacy
from spacy.pipeline import SentenceSegmenter

nlp_de = spacy.load('de_core_news_sm')

doc = nlp_de(u'Dies ist der erste Satz. Dies ist ein anderer Satz. Und ein dritter Satz.')

# einzelne Sätze ausgeben
doc_sents = [sent for sent in doc.sents]
print(doc_sents[1])

doc = nlp_de(u'Managment ist die Sachen richtig zu machen - Führung ist die richtigen Sachen zu machen. - Peter Drucker')
for sent in doc.sents:
    print(sent)

print(nlp_de.pipe_names)
# Pipeline anpassen

def set_custom_boundaries(doc):  # nach - beginnt ein neuer Satz
    for token in doc[:-1]:
        if token.text == '-':
            doc[token.i+1].is_sent_start = True
    return doc

nlp_de.add_pipe(set_custom_boundaries, before='parser')
print(nlp_de.pipe_names)

# Ausgabe
doc = nlp_de(u'Managment ist die Sachen richtig zu machen - Führung ist die richtigen Sachen zu machen. - Peter Drucker')
for sent in doc.sents:
    print(sent)

# splittet an \n
def split_on_new_lines(doc):
    start = 0
    seen_newline = False

    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'):
            seen_newline = True
    yield doc[start:]

sbd = SentenceSegmenter(nlp_de.vocab, strategy=split_on_new_lines)
nlp_de.add_pipe(sbd)
print(nlp_de.pipe_names)

doc = nlp_de(u'Dies ist ein Satz. Dies ist ein anderer.\n\nDies ist ein \ndritter Satz.')

for sent in doc.sents:
    print(sent)