import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import os
import csv


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width/dpi, height/dpi), dpi=dpi)
    matplotlib.use(orig_backend)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False)
    return anim.to_html5_video()
    pass


def plot_learning_curve(x, scores, figure_file):
    path = os.path.split(figure_file)
    # os.makedirs(os.path.join(*path[:-1]))
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(f'Running average of previous 100 scores {figure_file[:-12]}')
    plt.savefig(figure_file)
    with open(f"{figure_file[:-3]}csv", "w+") as file:
        data_backup = csv.writer(file)
        data_backup.writerow(scores)
