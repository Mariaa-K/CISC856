import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def plot_learning_curve(x, scores, figure_file):
    path = os.path.split(figure_file)
    # os.makedirs(os.path.join(*path[:-1]))
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    file = open(f"{figure_file[:-3]}csv", "w+")
    data_backup = csv.writer(file)
    data_backup.writerows(scores)
