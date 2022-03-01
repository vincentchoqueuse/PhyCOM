import numpy as np


if __name__ == '__main__':

    M = 100000
    SNR = 30
    for name in ["train", "test"]:
        filename = "{}.csv".format(name)
        seed_list = np.random.randint(low=0, high=100000000, size=M)
        data = np.transpose(np.vstack([seed_list]))
        np.savetxt(filename, data, delimiter=",", fmt="%d")
