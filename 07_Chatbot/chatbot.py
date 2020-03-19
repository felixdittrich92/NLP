# TF issue doesnt work !

import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM, Embedding

with open('./TextFiles/train_qa.txt', 'rb') as fp:
    train_data = pickle.load(fp)

with open('./TextFiles/test_qa.txt', 'rb') as fp:
    test_data = pickle.load(fp)

all_data = test_data + train_data
vocab = set()

for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')

vocab_len = len(vocab) + 1

max_story_len = max([len(data[0]) for data in all_data])
max_question_len = max([len(data[1]) for data in all_data])

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

train_story_text = list()
train_question_text = list()
train_answers = list()

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

# Texte vektorisieren
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_question_len=max_question_len):
    # Geschichten
    X = list()
    # Frage
    Xq = list()
    # Antwort
    Y = list()

    for story, query, answer in data:
        x = [word_index[word.lower()] for word in story]

        xq = [word_index[word.lower()] for word in query]

        # Array aus 0en 
        y = np.zeros(len(word_index)+1)

        # Yes/No setzt richtigen Wert auf 1
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

        return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

# PLACEHOLDER form = (max_story_len, batch_size)
input_sequence = Input((max_story_len, ))
question = Input((max_question_len, ))
vocab_size = len(vocab) + 1

# siehe Paper
# Encoder M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))

# Encoder C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))

# Question Encoder
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len))
question_encoder.add(Dropout(0.3))

# ENCODED <-- ENCODER(Input)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

# Modell zusammenfügen
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Modell trainieren
history = model.fit([inputs_train, queries_train], answers_train, 
                    batch_size=32, epochs=120, 
                    validation_data=([inputs_test, queries_test], answers_test))     

# Model speichern
model.save('07_Chatbot/chatbot_120_epochs.h5')  

""" Ergebnis plotten

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""

# eigene Storys predicten
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()
my_question = "Is the football in the garden ?"
mydata = [(my_story.split(), my_question.split(), 'yes')]
my_story, my_ques, my_ans = vectorize_stories(mydata)

pred_results = model.predict(([my_story, my_ques]))
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        print(k)

print(pred_results[0][val_max])




