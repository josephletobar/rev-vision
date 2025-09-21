import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import UnivariateSpline

def visual(file_path):
    pts = np.genfromtxt(file_path, delimiter=",", names=True)
    xs, ys = pts["x"], pts["y"]
    xs, ys = smooth(xs, ys)

    # Make a scatter/line plot
    plt.figure(figsize=(1, 5))
    plt.scatter(xs, ys, color="black")
    # plt.plot(xs, ys, color="red")

    plt.axis("off")
    plt.gca().invert_yaxis()  # flip y-axis to match image coordinates
    plt.show()

# TODO
def smooth(xs, ys, s=200):
    return xs, ys

# Testing   
if __name__ == "__main__":
    visual("output/points.csv")
    visual("examples/points_run.csv")