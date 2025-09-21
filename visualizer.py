import numpy as np
import matplotlib.pyplot as plt

pts = np.loadtxt("points.csv", delimiter=",")
xs, ys = pts[:, 0], pts[:, 1]

# make a scatter/line plot
plt.figure()
plt.plot(xs, ys)
plt.title("Ball Trail")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()  # flip y-axis to match image coordinates
plt.show()