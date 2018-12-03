# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Orthogonal

plt.style.use("ggplot")

class NeuralNet(Orthogonal.NeuralNet):
    def getBasis(self, n):
        Q,R = np.linalg.qr(np.random.randn(n,n))
        # I want the length of all columns to be 1/sqrt(n)
        columns = []
        for i in range(n):
            columns.append(1/np.sqrt(n) * Q.T[i])
        columns = np.array(columns)
        return columns.T
