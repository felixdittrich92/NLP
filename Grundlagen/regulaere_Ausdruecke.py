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