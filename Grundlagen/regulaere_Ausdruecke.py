import re

text = "Die Telefonnummer des Vertreters ist 040-555-1234. Ruf schnell an !"

# suchen ob Schlagwort in Text vorhanden
pattern = 'Telefonnummer'   # Muster
match = re.search(pattern, text)
print(match.span())
print(match.start())
print(match.end())

# alle Vorkommen in Text finden
text2 = "meine Telefonnummer ist eine neue Telefonnummer"
matches = re.findall(pattern, text2)
print(matches)
print(len(matches))

# alle Vorkommen inkl. Position ausgeben
for match in re.finditer(pattern, text2):
    print(match)

# Pattern mithilfe von Quantifizierern bilden
pattern = r'\d\d\d-\d\d\d-\d\d\d\d'  # siehe Tabelle Quantifiers (d - any number 0-9) 
phone = re.search(pattern, text)
print(phone.group())

pattern = r'\d{3}-\d{3}-\d{4}'
match = re.search(pattern, text)
print(match)

# gruppieren
print(phone.group())
pattern = r'(\d{3})-(\d{3})-(\d{4})'
phone = re.search(pattern, text)
print(phone.group(2))

# ODER Suche
result = re.search(r'Mann|Frau', "Dieser Mann war hier.")
print(result)

# nach Endung suchen
result = re.findall(r'.und', "Der Hund mit dem großen Mund war gesund.")
print(result)
# Leerzeichen nicht beachten // siehe Doku für weitere "Kürzungen"
result = re.findall(r'\S+und', "Der Hund mit dem großen Mund war gesund.")
print(result)
# Ausschluss von Zeichen
text = "Es sind 3 Nummern 34 in 5 diesem Satz."
result = re.findall(r'[^\d]+', text)
print(result)
# Satzzeichen entfernen
text = "Dies ist ein String! Aber er hat Satzzeichen. Wie können wir sie entfernen ?"
result = re.findall(r'[^!.?]+', text)
print(result)
# gruppieren direkt
text = "Finde nur die mit Binde-Strich getrennten Wörter in diesem Satz. Du weißt aber nicht, wie um-fassend sie sind."
result = re.findall(r'[\w]+-[\w]+', text)
print(result)
