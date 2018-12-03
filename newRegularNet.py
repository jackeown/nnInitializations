# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import code

plt.style.use("ggplot")

class NeuralNet:
    class Model(nn.Module):
        def __init__(self, inDim, outDim, **_settings):
            super().__init__()

            settings = {"nHiddenLayers": 1,"numNeurons":1}
            settings.update(_settings);

            layerDims = [inDim] + [settings["numNeurons"]]*settings["nHiddenLayers"] + [outDim]
            if "layerDims" in settings:
                layerDims = settings["layerDims"]

            self.layers = [nn.Linear(layerDims[i],layerDims[i+1]) for i in range(len(layerDims)-1)]
            self.setParameters()

        def setParameters(self):
            self.myParameters = []
            for layer in self.layers:
                self.myParameters.extend(layer.parameters())
            self.myParameters = nn.ParameterList(self.myParameters)

        def forward(self, x):
            x = torch.tensor(x, dtype = torch.float)
            for linearLayer in self.layers[:-1]:
                x = nn.functional.relu(linearLayer(x))
            return self.layers[-1](x)


    def __init__(self, inDim, outDim, debug=True, **_settings):
        self.debug = debug

        self.inDim = inDim
        self.outDim = outDim

        self.net = self.Model(inDim,outDim, **_settings) # make model.
        self.batchSize = 32
        self.learningRate = 0.01
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learningRate)

        # arrays to hold loss values over time for graphing.
        self.epochLosses = {"training":[], "validation": []}
        self.batchLosses = {"training":[], "validation": []}


    def fit(self, x, y, epochs = 10000, plotWhileTraining = True, valFrac = 0.1):
        # This function is more efficient if you
        # invoke it with large epoch sizes...
        x = torch.tensor(x, dtype = torch.float)
        y = torch.tensor(y, dtype = torch.float)

        n = len(x)
        trainN = int((1-valFrac)*n)
        indices = np.random.choice(n, trainN, replace=False)
        indicesComp = np.array(list(set(range(n)) - set(indices))) # set complement of indices
        self.trainX = x[indices]
        self.trainY = y[indices]
        self.valX = x[indicesComp]
        self.valY = y[indicesComp]

        batchesPerEpoch = trainN // self.batchSize

        if plotWhileTraining:
            fig,ax = plt.subplots()
            ax.autoscale(True)
            trainLine, = ax.plot(self.epochLosses["training"], color='green')
            valLine, = ax.plot(self.epochLosses["validation"], color='red')
            plt.ion()

        for e in range(epochs):
            for batch in range(batchesPerEpoch):
                indices = np.random.choice(trainN,trainN)
                # indices = np.random.choice(trainN,self.batchSize)
                x = torch.tensor(self.trainX[indices], dtype = torch.float)
                y = torch.tensor(self.trainY[indices], dtype = torch.float)
                self.optimizer.zero_grad()

                lossF = torch.nn.MSELoss()
                trainLoss = lossF(self.net(x),y)
                validationLoss = lossF(self.net(self.valX),self.valY)

                trainLoss.backward()
                self.optimizer.step()
                self.batchLosses["training"].append(float(trainLoss.data.cpu().numpy()))
                self.batchLosses["validation"].append(float(validationLoss.data.cpu().numpy()))

            for s in ["training", "validation"]:
                epochLoss = np.mean(self.batchLosses[s][-batchesPerEpoch:])
                self.epochLosses[s].append(epochLoss)

            if plotWhileTraining:
                limit = 100
                trainLine.set_data(np.arange(e+1),self.epochLosses["training"])
                valLine.set_data(np.arange(e+1),self.epochLosses["validation"])
                if e == epochs-1:
                    ax.set_xlim(0,e+1)
                    ax.set_ylim(0,max(self.epochLosses["validation"])*1.4)
                else:
                    ax.set_xlim(max(0,e-limit),e+1)
                    ax.set_ylim(0,max(self.epochLosses["validation"][-limit:])*1.4)
                plt.show()
                plt.pause(0.0001)

            if self.debug and e % 20 == 0:
                print("Training Loss for epoch {}: {}".format(e, self.epochLosses["training"][-1]))
                print("Validation Loss for epoch {}: {}".format(e, self.epochLosses["validation"][-1]))
                # if plotWhileTraining:
                #     plt.pause(0.0001)


    def predict(self,x):
        pred = self.net(x)
        return pred.cpu().detach().numpy().copy()
