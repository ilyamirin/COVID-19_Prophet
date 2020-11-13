import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
from tqdm import tqdm
import numpy
import sys

plt.rcParams['figure.figsize'] = [19.20, 10.80]
numpy.random.seed(1488)

data_file_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(data_file_path)

graph_data = list(zip(data.iloc[0:, 4:data.iloc[0:1].size - 1].values.flatten(), data.iloc[0:, 5:].values.flatten()))
#graph_data = list(filter(lambda x: x != (0, 0), set(graph_data)))

G = nx.DiGraph()
limit = len(graph_data)
for i in tqdm(range(int(sys.argv[1]), limit)):
    edge = graph_data[i:i+1][0]
    G.add_edge(edge[0], edge[1])
    nx.draw(G, node_size=5)
    plt.savefig(os.sep.join(["C:", "Users", "Admin", "Downloads", "covid-19", str(i) + ".png"]), dpi=100, orientation='landscape')
    plt.clf()
