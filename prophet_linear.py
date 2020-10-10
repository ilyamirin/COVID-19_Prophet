import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model


def print_list(lst: list):
    for element in lst:
        print(element)


def predict(data_path: str, days: int) -> list:
    data = pd.read_csv(data_path)

    mean = data.iloc[0:, 4:170].values.mean()
    data_x = data.iloc[0:, 4:169].values / mean
    data_y = data.iloc[0:, 5:170].values / mean

    reg = linear_model.BayesianRidge()
    reg.fit(list(map(lambda x: [x], data_x.flatten())), data_y.flatten())

    print((reg.predict([[969]]) - 981) / 981)
    print((reg.predict([[922853]]) - 927745) / 927745)

    s = 220
    result = list()
    for i in range(0, days):
        n = reg.predict([[s]])[0]
        result.append(int(n))
        s = n

    return result


base_path = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
confirmed_data_path = base_path + 'time_series_covid19_confirmed_global.csv'
recovered_data_path = base_path + 'time_series_covid19_recovered_global.csv'

confirmed = predict(confirmed_data_path, 100)
recovered = predict(recovered_data_path, 100)

print_list(confirmed)
print('-----------------------------------------------')
print_list(recovered)

