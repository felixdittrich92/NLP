import spacy
from spacy.matcher import Matcher

nlp_en = spacy.load('en_core_web_sm')

matcher = Matcher(nlp_en.vocab)

# Matcher hinzufügen/bauen siehe spacy Doku 
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

doc = nlp_en(u'The Solar Power industry continues to grow as demand for solarpower increases. Solar-power cars are popularity.')

# Treffer in Liste
matches = matcher(doc)

# Anzeige
for match_id, start, end in matches:
    string_id = nlp_en.vocab.strings[match_id]  # Stringrepräsentation
    span = doc[start:end]                       # Abschnitt des Matches
    print(match_id, string_id, start, end, span.text)

# Pattern entfernen aus Matcher
#matcher.remove('SolarPower')
