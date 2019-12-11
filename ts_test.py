from scipy.stats import t, norm
import numpy as np
import matplotlib.pyplot as plt


TRUE_MEAN = 2
TRUE_STD = 3
POP = [2]

MU = 2
V = 1
ALPHA = 1 / 2
BETA = 0


def update(x):
    global MU
    global V
    global ALPHA
    global BETA

    MU = (V * MU + x) / (V + 1)
    V += 1
    ALPHA += 1 / 2
    BETA += V * ((x - MU) ** 2) / (V + 1) / 2


def visualize():
    x = np.linspace(-3, 3, 100) * np.sqrt(BETA * (V + 1) / V / ALPHA) + MU

    # plt.hist(POP, density=True, alpha=0.2)
    plt.plot(x, t.pdf(x, df=ALPHA * 2, loc=MU,
                      scale=np.sqrt(BETA * (V + 1) / V / ALPHA)),
             label='Estimated pdf')
    plt.plot(x, norm.pdf(x, loc=TRUE_MEAN, scale=TRUE_STD),
             label='True pdf')
    plt.hist(
        np.random.standard_t(ALPHA * 2, size=100) * np.sqrt(BETA * (V + 1) / V / ALPHA) + MU,
        density=True, alpha=0.2
    )

    plt.legend()

    plt.show()


if __name__ == '__main__':
    for i in range(5):
        for _ in range(20):
            next_sample = np.random.normal(loc=TRUE_MEAN, scale=TRUE_STD)
            POP.append(next_sample)
            update(next_sample)

        visualize()
