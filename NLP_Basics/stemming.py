import nltk #gibt es in spaCy nicht / Wortstamm bestimmen
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# funktioniert nicht gut - PorterStemmer
print('--------Porter-Stemmer--------')
p_stemmer = PorterStemmer()

words = ['run', 'runner', 'ran', 'runs', 'fairly', 'trees', 'numbers', 'feeds']

for word in words:
    print(word +  '--->' + p_stemmer.stem(word))

# funktioniert etwas besser - SnowballStemmer
print('--------Snowball-Stemmer--------')
s_stemmer = SnowballStemmer(language='english')

for word in words:
    print(word +  '--->' + s_stemmer.stem(word))

# geht auch für deutsch allerdings noch schlechter !! 