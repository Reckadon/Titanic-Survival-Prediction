import matplotlib.pylab as plt

def plot(X, Y, x_label, y_label, size: tuple[int, int]):
    plt.figure(figsize=size)
    plt.plot(X, Y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()