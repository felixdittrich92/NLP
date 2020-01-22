import spacy

nlp = spacy.load('en_core_web_md')

print(nlp(u'lion').vector.shape)
print(nlp(u'The quick brown fox jumped').vector.shape)


# einzelne Wörter auf Gleichheit bzw. Zusammenhang überprüfen
tokens = nlp(u'lion cat pet')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print(len(nlp.vocab.vectors))

tokens = nlp(u'dog cat nargle')

for token in tokens:
    #        Text      ist vorhanden    vector als "Wert"   ist nicht vorhanden
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)

