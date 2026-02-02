import numpy as np
import matplotlib.pyplot as plt


def drift_fault(ts, m, M, K, b):
    """
    ts: time series
    m: start index
    M: number of steps it scales for
    K: number of steps we get constant error after
    b: bias
    """

    ts[m: m+ M] += b * np.linspace(0, M) / M

    ts[m+M: m+M+K] += b
    return ts


def noise_fault(ts,m, M, c):
    """
    ts: time series
    m: start index
    M: m + + is end index
    c: std
    """

    ts[m: m+ M] += np.random.normal(0, c, size=M)

    return ts




if __name__ == "__main__":
    ts = np.sin(np.linspace(0, 50, 500))
    ts_noise_fault = noise_fault(ts.copy(), m=100, M=50, c=0.1)
    plt.plot(ts, label="Original")
    plt.plot(ts_noise_fault, label="Noise Fault")
    plt.xlim(100 - 20, 100 + 50 + 20)
    plt.legend()
    plt.show()

    ts_drift_fault = drift_fault(ts.copy(), m=300, M=50, K=50, b=0.5)
    plt.plot(ts, label="Original")
    plt.plot(ts_drift_fault, label="Drift Fault")
    plt.xlim(300 - 20, 300 + 50 + 50 + 20)
    plt.legend()
    plt.show()