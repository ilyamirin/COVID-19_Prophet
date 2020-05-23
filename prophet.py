from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

data_file_path = '/Users/ilyamirin/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(data_file_path)

train_data_x = data.iloc[0:, 4:120].values
train_data_y = (data.iloc[0:, 5:121].values - train_data_x) / (train_data_x + 0.00000001)

model = Sequential()
model.add(Dense(1024, input_shape=(1,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data_x.flatten(), train_data_y.flatten(), epochs=150)
