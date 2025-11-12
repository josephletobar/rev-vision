import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visual(file_path):
    # Load points
    pts = np.genfromtxt(file_path, delimiter=",", names=True)
    xs, ys = pts["x"], pts["y"]

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], 'bo-', lw=2)  # blue dots with connecting line

    ax.set_xlim(np.min(xs) - 10, np.max(xs) + 10)
    ax.set_ylim(np.min(ys) - 10, np.max(ys) + 10)
    ax.invert_yaxis()

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(xs[:frame], ys[:frame])
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=10,
        blit=False,
        repeat=False
    )

    plt.show()

if __name__ == "__main__":
    visual("outputs/points.csv")