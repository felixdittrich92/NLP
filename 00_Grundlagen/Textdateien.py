from datetime import datetime

# Strings verbinden
person = "Felix"
print(f'Mein Name ist {person}')
print(f'Mein Name ist {person!r}')

dict = {'a': 1, 'b': 2}
print(f"Meine Zahl ist: {dict['a']}")

list = [1, 2, 3]
print(f"Meine Zahl ist: {list[2]}")

library = [('Autor', 'Thema', 'Seiten'), ('Twain', 'Rafting im Wasser', '401'),
            ('Twain', 'Rafting', '601'), ('Feyman', 'Physik', '95'),
            ('Hamilton', 'Mythologie', '144')]

for autor, thema, seiten in library:
    print(f'{autor:{10}} {thema:{20}} {seiten:>{10}}')  # Überschrift: < links > rechts ^ mittig // vor dem Zeichen - oder . füllt übrigen Zeichen damit auf

heute = datetime(year=2019, month=12, day=27)
print(f'{heute:%d. %B %Y}')

# Textdateien erstellen/einlesen/lesen
textfile = open("test.txt", 'w') #w+ schreiben + lesen

textfile.write("Hallo, dies ist eine kleine Textdatei.\n")
textfile.write("Dies ist eine zweite Zeile")

textfile.close()

text = open("test.txt")
print(text.read())

#for line in text:
#    print(line)

text.seek(0) # Text zurücksetzen
print(text.readlines())

text.close()

#with open("test.txt", 'r') as txt:
#    erste_zeile = txt.readlines()