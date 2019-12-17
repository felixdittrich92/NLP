import spacy
from spacy import displacy

nlp_de = spacy.load('de_core_news_sm')

# Beispiele Strings in Tokens teilen 
mystring = '"Wir ziehen nach L.A.!"'

doc = nlp_de(mystring)

for token in doc:
    print(token.text)

print('-----------------------------')
doc2 = nlp_de(u'Wir sind hier zum Helfen! Online-Support per Post, E-mail an support@email.com oder besuche uns auf der Homepage: http://www.unsere_seite.de!')

for token in doc2:
    print(token.text)

print('-----------------------------')
doc3 = nlp_de(u'Ein 5km Taxi-Trip in Berlin kostet ca. €10.30!')

for token in doc3:
    print(token.text)

print('-----------------------------')
doc4 = nlp_de(u'Lass uns St. Louis in den U.S.A. im nächsten Jahr ge-mein-sam besuchen D-U_D-E.')

for token in doc4:
    print(token.text)
print(len(doc4))
print(len(doc4.vocab)) # Vokabular welches in dem geladenen nlp_de vorhanden ist. Gibt noch wesentlich größere

print('-----------------------------')
doc5 = nlp_de(u'Es ist besser zu geben, denn zu nehmen.')

# Position
print(doc5[0])
# von Position x1 bis x2
print(doc5[3:5])
# letzten 5 bis Ende
print(doc5[-5:])
# letzte Wort / Sonderzeichen
print(doc5[-1])

# Englisch laden
nlp_en = spacy.load('en_core_web_sm')

print('-----------------------------')
doc6 = nlp_de(u'BMW baut eine neue Fabrik in Bad Reichenhall für $6 Millionen')

# Entitäten ausgeben
for ent in doc6.ents:
    print(ent.text) 
    print(ent.label_) 
    print(spacy.explain(ent.label_))

print('-----------------------------')
doc7 = nlp_de(u'Autonome Fahrzeuge verlagern die Versicherungsverantwortung zum Hersteller. Der grüne, alte, rostende Mercedes fährt um die Ecke.')

# gibt zusammenhängende Strings aus 
for chunk in doc7.noun_chunks:
    print(chunk.text)

# displaCy - Visualisierung
print('-----------------------------')
doc8 = nlp_de(u'BMW baut eine neue Fabrik in Bad Reichenhall für $6 Millionen')
displacy.serve(doc8, style='dep', options={'distance':80}) # jupyter=True 
displacy.serve(doc8, style='ent', options={'distance':80})