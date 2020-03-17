import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

iris = load_iris()

y = iris.target
X = iris.data
print(f'Before\n{y}')

y = to_categorical(y)
print(f'After\n{y}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

scaler_object = MinMaxScaler() # alle Werte aus Matrix durch höchsten Wert 
scaled_X_train = scaler_object.fit_transform(X_train)
scaled_X_test = scaler_object.fit_transform(X_test)

model = Sequential()

model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(scaled_X_train, y_train, epochs=150, verbose=2)

result = model.predict_classes(scaled_X_train)
print(result)
print(model.metrics_names)
print(model.evaluate(x=scaled_X_test, y=y_test))

model.save('mein_model.h5')

new_model = load_model('mein_model.h5')
result = new_model.predict_classes(X_test)
print(result)

