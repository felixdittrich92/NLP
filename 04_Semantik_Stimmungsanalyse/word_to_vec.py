import spacy
from scipy import spatial

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

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

# king - men + woman  --> NEW vector: queen, princes
king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

new_vector = king - man - woman

computed_similarity = list()

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarity.append((word, similarity))

computed_similarities = sorted(computed_similarity, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])
