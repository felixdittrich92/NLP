import spacy
import numpy as np
import random
import os

from pickle import dump, load

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all Tensorflow messages

nlp = spacy.load('de', disable=['parser', 'tagger', 'ner']) # disable wird nicht benötigt
nlp.max_length = 336367 # maximalen Wörter

# Text einlesen
def read_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        str_text = f.read()

    return str_text

# Text bereinigen   
def seperate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text.strip() not in '\ufeff  \\ufeff \\n\\n \\n\\n\\n!\"-#$%&()--.*+,-/:;<=>?@[\\\\]^_`{|}~\\t\\n « »']

d = read_file('./TextFiles/hesse_unterm_rad_vier_kapitel.txt')
tokens = seperate_punc(d)

# Sequenzen des Textes erzeugen
train_len = 25+1 

text_sequences = list()

for i in range(train_len, len(tokens)):
    seq = tokens[i - train_len : i]
    text_sequences.append(seq)

print(' '.join(text_sequences[0]))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

# IDs der Wörter
for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')

# Größe des Vokabulars
vocabulary_size = len(tokenizer.word_counts)

# Sequenzen in Arrays umwandeln
sequences = np.array(sequences)

# Features
X = sequences[:,:-1]
# Labels
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocabulary_size+1)

# Anzahl der Wörter in einer Sequenz -> 25
seq_len = X.shape[1]

# Modell erstellen
def create_model(vocabulary_size, seq_len):
    model = Sequential()

    # input, output, Länge der Sequenzen
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len)) # wandelt die positiven Integer aus dem Array in Vektoren um
    model.add(LSTM(seq_len*2, return_sequences=True))
    model.add(LSTM(seq_len*2))
    model.add(Dense(seq_len*2, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax')) # softmax 1 oder 0

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Zusammenfassung des Models
    model.summary()

    return model

model = create_model(vocabulary_size+1, seq_len)
model.fit(X, y, batch_size=128, epochs=50, verbose=1)

model.save('06_DeepLearning_NLP/mein_neues_model.h5')
dump(tokenizer, open('06_DeepLearning_NLP/mein_tokenizer', 'wb'))

# Text generieren
def generate_text(model, tokenizer, seq_len, seed_text, num_gen_text):
    output_text = list()

    input_text = seed_text

    for _ in range(num_gen_text):

        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre') # pre füllt Felder vor dem String aus
        
        pred_word_ind = model.predict(pad_encoded, verbose=0)[0]
        
        pred_word = tokenizer.index_word[pred_word_ind]
        
        input_text += ' ' + pred_word

        output_text.append(pred_word)

    return ' '.join(output_text)


random.seed(101)
random_pick = random.randint(0, len(text_sequences))
random_seed_text = text_sequences[random_pick]
seed_text = ' '.join(random_seed_text)

generate_text(model, tokenizer, seq_len, seed_text, 25)
