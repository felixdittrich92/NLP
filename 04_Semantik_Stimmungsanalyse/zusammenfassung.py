import spacy
from scipy import spatial
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Part1: Führe eine Vektorarithmetik aus und vergleiche die Wörter

nlp = spacy.load('en_core_web_md')

# Wähle Wörter zum vergleichen und beziehe ihre Vektoren
word1 = nlp.vocab['wolf'].vector
word2 = nlp.vocab['dog'].vector
word3 = nlp.vocab['cat'].vector

# Schreibe eine Funktion um die Distanz zwischen den Vektoren zu berechnen
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
new_vector = word1 - word2 + word3

# Gib die 10 besten Treffer in Bezug zu dem Vokabular aus
computed_similarities = list()

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                sim = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, sim))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
print([w[0].text for w in computed_similarities[:10]])

# Schreibe eine Funktion die 3 Strings annimmt a-b+c Arithmetik ausführt und die 10 besten Treffer zurück gibt
def vector_math(a,b,c):
    word1 = nlp.vocab[a].vector
    word2 = nlp.vocab[b].vector
    word3 = nlp.vocab[c].vector

    new_vector = word1 - word2 + word3

    computed_similarities = list()

    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    sim = cosine_similarity(new_vector, word.vector)
                    computed_similarities.append((word, sim))

    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
    return [w[0].text for w in computed_similarities[:10]]

# teste die Funktion
sim_words = vector_math('king', 'man', 'woman')
print(sim_words)


# Part2: Führe eine Vader Sentimentanalyse aus
sid = SentimentIntensityAnalyzer()

review = 'This movie is really really bad.'

print(sid.polarity_scores(review))

# Schreibe eine Funktion die einen String annimmt und zurück gibt ob dieser positiv, negativ oder neutral ist
def review_rating(string):
    scores = sid.polarity_scores(string)
    if scores['compound'] == 0:
        return 'Neutral'
    elif scores['compound'] > 0:
        return 'Positive'
    else:
        return 'Negative'

# Teste die Funktion
result = review_rating(review)
print(result)
