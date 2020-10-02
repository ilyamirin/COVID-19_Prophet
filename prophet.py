from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

data_file_path = '~/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(data_file_path)

mean = data.iloc[0:, 4:100].values.mean()
data_x = data.iloc[0:, 4:99].values / mean
data_y = data.iloc[0:, 5:100].values / mean

model = Sequential()
model.add(Dense(1024, input_shape=(1,), activation='relu', kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mse', optimizer='adam')

best_gen = (0, 1000000)
for i in range(0, 5):
    history = model.fit(data_x[:, :-10].flatten(), data_y[:, :-10].flatten(), epochs=64, verbose=1, batch_size=1024)
    mse = model.evaluate(data_x[:, -10:].flatten(), data_y[:, -10:].flatten())
    if mse < best_gen[1]:
        best_gen = (i, mse)
        model.save("./model.h5")

model1 = load_model("model.h5")

#retro
real_date = data.iloc[:, -1:].axes[1].values[0]
real = data.iloc[:, -1:].values
pred_date = data.iloc[:, -2:-1].axes[1].values[0]
pred = model1.predict(data.iloc[:, -2:-1].values).astype(int)
delta = (real - pred) / (real+0.0000000001) * 100

result_data = np.concatenate((data.iloc[0:, 1:2], data.iloc[0:, 0:1], pred, real, delta.astype(int)), axis=1)
result = pd.DataFrame(result_data, columns=['Страна', 'Провинция', 'Прогноз на %s' % real_date, 'Итог на %s' % real_date, 'Ошибка прогноза в %'])

result.to_csv('prediction.csv', sep=';')

#new one
pred = model1.predict(data.iloc[:, -1:].values).astype(int)
(pred[187] - real[187]) / 2

model1.predict(np.array([969])) #981
model1.predict(np.array([6053]))
model1.predict(np.array([326448]))
model1.predict(np.array([335882]))
model1.predict(np.array([922853])) #927745
model1.predict(np.array([1030690]))

s = 200
for i in range(0, 100):
    n = model1.predict(np.array([s]))[0][0]
    print(n)
    s = n

