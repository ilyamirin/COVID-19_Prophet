from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file_path = '~/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(data_file_path)

mean = data.iloc[0:, 4:].values.mean()
data_x = data.iloc[0:, 4:-1].values / mean
data_y = data.iloc[0:, 5:].values / mean

model = Sequential()
model.add(Dense(512, input_shape=(1,), activation='relu', kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mse', optimizer='adam')

best_gen = (0, 1000000)
for i in range(0, 10):
    history = model.fit(data_x[:, :-10].flatten(), data_y[:, :-10].flatten(), epochs=64, verbose=0)
    mse = model.evaluate(data_x[:, -10:].flatten(), data_y[:, -10:].flatten())
    if mse < best_gen[1]:
        best_gen = (i, mse)
        model.save("./model.h5")
    print(i, mse, model.predict(np.array([969])), model.predict(np.array([326448])))

model1 = load_model("model.h5")

print(model.predict(data.iloc[187:188, -1:].values[0])[0][0])

model1.predict(np.array([969])) #981
model1.predict(np.array([6053]))
model1.predict(np.array([326448]))
model1.predict(np.array([335882]))

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.show()

