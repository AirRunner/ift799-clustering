from math import comb
import numpy as np
import pandas as pd


def rand(result, original):
    n = len(result)
    c = len(np.unique(result))
    k = len(np.unique(original))

    contingence = pd.crosstab(result, original).to_numpy()

    n11 = 0
    for i in range(c):
        for j in range(k):
            n11 += comb(contingence[i, j], 2)

    n10 = -n11
    for j in range(k):
        n10 += comb(contingence[:, j].sum(), 2)

    n01 = -n11
    for i in range(c):
        n01 += comb(contingence[i, :].sum(), 2)

    n00 = comb(n, 2) - (n11 + n10 + n01)

    return (n11 + n00) / (n01 + n10 + n11 + n00)
