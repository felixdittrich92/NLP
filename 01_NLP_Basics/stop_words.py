import spacy

nlp_de = spacy.load('de_core_news_sm')

# Zeige alle Stoppwörter
print(nlp_de.Defaults.stop_words)
print(len(nlp_de.Defaults.stop_words))

# Zeige ob Wort Stopwort ist
print(nlp_de.vocab['das'].is_stop)
print(nlp_de.vocab['Häuser'].is_stop)

# Stopwortliste erweitern
nlp_de.Defaults.stop_words.add('bspw')
nlp_de.vocab['bspw'].is_stop = True

# ein Wort mehr durch hinzufügen
print(len(nlp_de.Defaults.stop_words))

# Stopwort aus Liste entfernen
nlp_de.Defaults.stop_words.remove('wir')
nlp_de.vocab['wir'].is_stop = False