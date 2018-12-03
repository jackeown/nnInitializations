# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Orthogonal
from scipy.linalg import hadamard

plt.style.use("ggplot")

class NeuralNet(Orthogonal.NeuralNet):

    def getBasis(self, n):
        Q = np.random.uniform(0,2/np.sqrt(n),(n,n))
        p = int(2**np.ceil(np.log2(n)))
        H = hadamard(p)
        selectedRowsOfH = np.random.choice(p,n,replace=False)
        H = H[selectedRowsOfH][:,:n]

        columns = []
        for i in range(n):
            column = Q.T[i]
            column *= H.T[i]
            columns.append(column)
        columns = np.array(columns)
        return columns.T
