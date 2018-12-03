# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import convNet

plt.style.use("ggplot")


class NeuralNet(convNet.NeuralNet):

    class Model(convNet.NeuralNet.Model):
        def getBasis(self, n):
            Q, R = np.linalg.qr( np.random.normal(0,1,(n,n)) )
            # Q is orthonormal, but I want just orthogonal with
            # mean magnitude: 1/sqrt(n), std_deviation: ?
            # solution: multiply columns by random scalars.
            columns = []
            for i in range(n):
                columns.append(np.random.uniform(0,1)/np.sqrt(n) * Q.T[i])
            columns = np.array(columns)
            return columns.T

        def getRandomConvWeight(self, shape, xavier = False):
            """Returns a random convolutional filter according to orthogonal
            weight-initialization if random == True, else returns a Xavier initialized
            random convolutional filter."""
            pass # I have no idea how to do this...

        def getRandomLinearWeight(self, shape, xavier = False):
            """Returns a random linear weight matrix according to orthogonal
            weight-initialization if random == True, else returns a Xavier initialized
            random linear weight matrix."""
            # shape is (100,32) for a weight matrix such that
            # Ax=y, where len(x) = 32 and len(y) = 100

            if xavier == True:
                return np.random.normal(0,1/np.sqrt(shape[1]), (shape[0],shape[1]))
            else:
                weight = self.getBasis(shape[1]) # will be a matrix of size (input x input)
                if shape[1] > shape[0]: # more inputs than outputs
                    # pick a random subset of these rows (weight vectors).
                    rowsToKeep = np.random.choice(shape[1], shape[0], replace=False)
                    weight = weight[rowsToKeep]
                elif shape[1] < shape[0]: # less inputs than outputs
                    # augment basis with xavier initialzed rows (weight vectors).
                    augShape = (shape[0] - shape[1], shape[1])
                    augment = np.random.normal(0,1/np.sqrt(shape[1]), augShape)
                    weight = np.concatenate((weight,augment),axis=0)
                return weight

        def __init__(self, inDim, outDim, **_settings):
            super().__init__(inDim,outDim, **_settings)
            settings = {"linearXavier":False, "convXavier":False}
            settings.update(_settings)

            for layer in self.linearLayers:
                shape = layer.weight.shape
                weight = self.getRandomLinearWeight(shape, xavier=settings["linearXavier"])
                layer.weight = nn.Parameter(torch.from_numpy(weight).float())

            self.setParameters()
