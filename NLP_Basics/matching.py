import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp_en = spacy.load('en_core_web_sm')

matcher = Matcher(nlp_en.vocab)

# Matcher hinzufügen/bauen siehe spacy Doku 
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]

matcher.add('SolarPowerMatcher', None, pattern1, pattern2, pattern3)

doc = nlp_en(u'The Solar Power industry continues to grow as demand for solarpower increases. Solar-power cars are popularity.')

# Treffer in Liste
matches = matcher(doc)

# Anzeige
for match_id, start, end in matches:
    string_id = nlp_en.vocab.strings[match_id]  # Stringrepräsentation
    span = doc[start:end]                       # Abschnitt des Matches -> doc[start-5:end+5] 5 Wörter vor und nach match mit ausgeben
    print(match_id, string_id, start, end, span.text)

# Pattern entfernen aus Matcher
#matcher.remove('SolarPower')

print('---------------------------------------------')
# Phrasenmatcher

matcher = PhraseMatcher(nlp_en.vocab)

with open('../TextFiles/reaganomics.txt', encoding='unicode_escape') as f:
    doc2 = nlp_en(f.read())

phrase_list = ['vodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']
phrase_patterns = [nlp_en(text) for text in phrase_list]
matcher.add('EconMatcher', None, *phrase_patterns) # * Aufsplittung der Parameter

matches = matcher(doc2)

# Anzeige
for match_id, start, end in matches:
    string_id = nlp_en.vocab.strings[match_id]  # Stringrepräsentation
    span = doc2[start:end]                       # Abschnitt des Matches
    print(match_id, string_id, start, end, span.text)
