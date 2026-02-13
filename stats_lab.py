import numpy as np
import matplotlib.pyplot as plt


# ================= Histogram =================

def normal_histogram(n):
    samples = np.random.normal(loc=0, scale=1, size=n)
    plt.hist(samples, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1)")
    plt.show()
    return samples


def uniform_histogram(n):
    samples = np.random.uniform(low=0, high=10, size=n)
    plt.hist(samples, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")
    plt.show()
    return samples


def bernoulli_histogram(n):
    samples = np.random.binomial(n=1, p=0.5, size=n)
    plt.hist(samples, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")
    plt.show()
    return samples


# ================= Mean & Variance =================

def sample_mean(data):
    data = np.asarray(data)
    return data.sum() / data.size


def sample_variance(data):
    data = np.asarray(data)
    m = sample_mean(data)
    return ((data - m) ** 2).sum() / (data.size - 1)   # n-1


# ================= Order Statistics =================

def order_statistics(data):
    arr = np.sort(np.asarray(data))

    mn = arr[0]
    mx = arr[-1]
    med = np.median(arr)

    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)

    return (mn, mx, med, q1, q3)


# ================= Sample Covariance =================

def sample_covariance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mx = sample_mean(x)
    my = sample_mean(y)

    return np.dot(x - mx, y - my) / (x.size - 1)


# ================= Covariance Matrix =================

def covariance_matrix(x, y):
    vx = sample_variance(x)
    vy = sample_variance(y)
    cxy = sample_covariance(x, y)

    return np.array([[vx, cxy],
                     [cxy, vy]])


# ================= Example Run =================

if __name__ == "__main__":

    d1 = normal_histogram(1000)
    d2 = uniform_histogram(1000)
    d3 = bernoulli_histogram(1000)

    print("Mean:", sample_mean(d1))
    print("Variance:", sample_variance(d1))

    test = np.array([5, 1, 3, 2, 4])
    print("Order Stats:", order_statistics(test))

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    print("Sample Covariance:", sample_covariance(x, y))
    print("Covariance Matrix:\n", covariance_matrix(x, y))
