import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json
import os

# Generate random 100 Rows
df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD')).to_numpy()
print(df)

np.random.seed(7)
# load pima indians dataset

# split into input (X) and output (Y) variables
X = df[:, 0:3]
Y = df[:, 3]
# create model
model = Sequential(
    [Dense(5, input_dim=3), Activation('relu'), Dense(8), Activation('relu'), Dense(1), Activation('softmax')])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


model.save("model.h5")