from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file_path = '/Users/ilyamirin/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(data_file_path)

data_x = data.iloc[0:, 4:-1].values / data.iloc[0:, 4:].values.mean()
data_y = data.iloc[0:, 5:].values / data.iloc[0:, 4:].values.mean()

model = Sequential()
model.add(Dense(1024, input_shape=(1,), activation='relu', kernel_initializer='normal'))
model.add(Dense(1, activation='relu', kernel_initializer='normal'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

history = model.fit(data_x[:, :-10].flatten(), data_y[:, :-10].flatten(), epochs=64)

model.predict(np.array([6053]))
model.predict(np.array([326448]))
model.predict(np.array([335882]))


model.evaluate(data_x[:, -10:].flatten(), data_y[:, -10:].flatten())

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
