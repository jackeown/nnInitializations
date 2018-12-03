# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("ggplot")

class NeuralNet:
    def __init__(self, inDim, outDim, debug=True):
        self.activation = torch.nn.functional.relu #used to be MyReLU.apply
        self.debug = debug
        self.dtype = torch.float
        self.device = torch.device("cuda:0") # Uncomment this to run on GPU
        # self.device = torch.device("cpu") # Uncomment this to run on CPU
        self.inDim = inDim
        self.outDim = outDim

        nHiddenLayers = 2
        numNeurons = 100
        self.layers = [inDim] + [numNeurons]*nHiddenLayers + [outDim]

        # self.layers = [inDim, 100, 100, 100, 100, 100, outDim]
        self.learningRate = 1e-3
        self.batchSize = 32

        self.weights = []
        self.biases = []
        # initialize weight and bias matrices as random normals of the appropriate shape.
        for i in range(1,len(self.layers)):
            weight = torch.randn(self.layers[i-1], self.layers[i])/np.sqrt(self.layers[i-1])
            bias = torch.randn(1, self.layers[i])/np.sqrt(self.layers[i-1])

            weight = torch.tensor(weight, dtype=self.dtype, device=self.device, requires_grad=True)
            bias = torch.tensor(bias, dtype=self.dtype, device=self.device, requires_grad=True)
            self.weights.append(weight)
            self.biases.append(bias)

        self.optimizer = torch.optim.Adam(self.weights + self.biases, lr = self.learningRate)

        # arrays to hold loss values over time for graphing.
        self.epochLosses = {"training":[], "validation": []}
        self.batchLosses = {"training":[], "validation": []}

    def forwardPass(self, x):
        x = torch.tensor(x, dtype = self.dtype, device = self.device)
        a = x
        for w,b in list(zip(self.weights, self.biases))[:-1]:
            a = self.activation(a.mm(w) + b)
        a = a.mm(self.weights[-1]) + self.biases[-1]
        return a


    def fit(self, x, y, epochs = 10000, plotWhileTraining = True, valFrac = 0.2):
        # This function is more efficient if you
        # invoke it with large epoch sizes...
        x = torch.tensor(x, dtype = self.dtype, device = self.device)
        y = torch.tensor(y, dtype = self.dtype, device = self.device)

        n = len(x)
        trainN = int((1-valFrac)*n)
        indices = np.random.choice(n, trainN, replace=False)
        indicesComp = np.array(list(set(range(n)) - set(indices))) # set complement of indices
        self.trainX = x[indices]
        self.trainY = y[indices]
        self.valX = x[indicesComp]
        self.valY = y[indicesComp]

        batchesPerEpoch = trainN//self.batchSize

        if plotWhileTraining:
            fig,ax = plt.subplots()
            ax.autoscale(True)
            trainLine, = ax.plot(self.epochLosses["training"])
            valLine, = ax.plot(self.epochLosses["validation"])
            plt.ion()


        for e in range(epochs):
            for batch in range(batchesPerEpoch):
                indices = np.random.choice(trainN,self.batchSize)
                x = torch.tensor(self.trainX[indices], dtype = self.dtype, device = self.device)
                y = torch.tensor(self.trainY[indices], dtype = self.dtype, device = self.device)
                self.optimizer.zero_grad()

                # Forward Pass...
                # pred = self.forwardPass(x)

                lossF = torch.nn.MSELoss()
                trainLoss = lossF(self.forwardPass(x),y)
                validationLoss = lossF(self.forwardPass(self.valX),self.valY)

                trainLoss.backward()
                self.optimizer.step()
                self.batchLosses["training"].append(float(trainLoss.data.cpu().numpy()))
                self.batchLosses["validation"].append(float(validationLoss.data.cpu().numpy()))

            for s in ["training", "validation"]:
                epochLoss = np.mean(self.batchLosses[s][-batchesPerEpoch:])
                self.epochLosses[s].append(epochLoss)

            if plotWhileTraining:
                limit = 100
                valLine.set_data(np.arange(e+1),self.epochLosses["validation"])
                trainLine.set_data(np.arange(e+1),self.epochLosses["training"])
                if e == epochs-1:
                    ax.set_xlim(0,e+1)
                    ax.set_ylim(0,max(self.epochLosses["validation"])*1.4)
                else:
                    ax.set_xlim(max(0,e-limit),e+1)
                    ax.set_ylim(0,max(self.epochLosses["validation"][-limit:])*1.4)
                plt.show()

            if self.debug and e % 20 == 0:
                print("Training Loss for epoch {}: {}".format(e, self.epochLosses["training"][-1]))
                print("Validation Loss for epoch {}: {}".format(e, self.epochLosses["validation"][-1]))
                plt.pause(0.0001)


    def predict(self,x):
        pred = self.forwardPass(x)
        return pred.cpu().detach().numpy().copy()
