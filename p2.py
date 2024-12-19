import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/Engineering/7th Sem/NN DL LAB/Programs syllabus-20241115/BankNote_Authentication.csv')
data = pd.DataFrame(dataset)
data.head()

data.replace('...', float('nan'), inplace=True)
data.dropna(inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

X = data.drop(['class'], axis=1)
Y = data['class']

model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.33, epochs=250, batch_size=10, verbose=1)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
