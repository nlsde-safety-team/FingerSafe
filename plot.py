import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D


def update_scatter(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([64])

def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(8)

groups = {'Clean': [[10, 15, 20, 25, 30, 100], [3.36, 16.04, 39.18, 61.32, 78.11, 95.14]],
          'PGD': [[10, 15, 20, 25, 30, 100], [2.86, 12.06, 24.88, 37.94, 52.86, 82.34]],
          'Unlearn.': [[10, 15, 20, 25, 30, 100], [2.74, 10.95, 24.75, 37.94, 56.09, 83.21]],
          'Fawkes': [[10, 15, 20, 25, 30, 100], [5.72, 18.03, 35.82, 49.63, 60.82, 76.87]],
          'Lowkey': [[10, 15, 20, 25, 30, 100], [3.11, 12.44, 23.51, 36.19, 51.62, 77.74]],
          '[35]': [[10, 15, 20, 25, 30, 100], [2.61, 11.57, 22.64, 39.43, 57.21, 59.33]],
          'FingerSafe': [[10, 15, 20, 25, 30, 100], [2.74, 4.60, 5.10, 3.86, 4.48, 5.85]]}


# plot setting
subplot_loc = [[0.055, 0.5], [0.305, 0.5], [0.555, 0.5], [0.805, 0.5],
               [0.055, 0.1], [0.305, 0.1], [0.555, 0.1], [0.805, 0.1]]
colors = ['deepskyblue', 'yellowgreen', 'violet', 'darkturquoise', 'gold', 'tomato', 'limegreen', 'deeppink', 'sandybrown', 'orange', 'cornflowerblue', 'lightsalmon', 'crimson', 'gray', 'darkcyan']
markers = ['^', 'v', 'P', 'X', 'o', 's', 'D', '*', 'h', '>', '<', 'd', 'p', 'H', '8']
mker_sz = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
linestyles = ['--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--']


# plot
plt.figure(figsize=(8, 5.6))
plt.style.use('seaborn-darkgrid')
# for i in range(0, 8):
#     plt.axes(subplot_loc[i] + [0.193, 0.27])
#     # plt.grid(b=True, linestyle='--', color='r')
#     for ind, (model_name, data) in enumerate(groups[i]):
#         plt.plot(data[list(data)[0]], data[list(data)[1]], c=colors[ind], marker=markers[ind], markersize=mker_sz[ind], linestyle=linestyles[ind], linewidth=1.0, label=model_name)
#     plt.xlabel(list(data)[0])
#     plt.ylabel(list(data)[1])
#     #plt.title("")
# plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter), plt.Line2D: HandlerLine2D(update_func=updateline)}, loc=(-4.18, 2.83), ncol=8, fontsize=12)
ind = 0
for model_name, data in groups.items():
    plt.plot(data[0], data[1], c=colors[ind], marker=markers[ind], markersize=mker_sz[ind], linestyle=linestyles[ind], linewidth=1.0, label=model_name)
    ind += 1
# plt.xlabel(list(data)[0])
# plt.ylabel(list(data)[1])
#plt.title("")
# plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter), plt.Line2D: HandlerLine2D(update_func=updateline)}, loc=(-4.18, 2.83), ncol=8, fontsize=12)
plt.xlabel("protection rate (%)")
plt.ylabel("naturalness")
plt.legend()
plt.show()
eps_fig = plt.gcf() # 'get current figure'
eps_fig.savefig('./result/naturalness.eps', format='eps', dpi=1000)
