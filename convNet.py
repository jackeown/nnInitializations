import code
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# from time import process_time as time
from tensorboardX import SummaryWriter
writer = SummaryWriter()

from initializations import getRandomWeightInit

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    labels = np.argmax(labels, axis=1)
    # return (outputs == labels)
    return np.sum(outputs==labels)/float(labels.size)

plt.style.use("ggplot")
class NeuralNet:

    class Model(nn.Module):
        def __init__(self, inDim, outDim, **_settings):
            super().__init__()

            # A convLayer is (in_channels, out_channels, kernel_size)
            self.settings = {"convLayers": []}
            self.settings.update(_settings);

            def shapeAfterConv(inDim,conv):
                """Returns the number of neurons after applying
                a convolutional layer to an input of shape 'inDim'"""

                # if kernel_size is an int, make it a tuple instead.
                if(type(conv[2]) == int):
                    conv = (conv[0],conv[1],(conv[2],conv[2]))

                return (conv[1],(inDim[1]-conv[2][0]+1),(inDim[2]-conv[2][1]+1))

            if type(inDim) == int:
                inDim = (1,inDim,1)

            self.convOutShape = inDim
            for i,_ in enumerate(self.settings["convLayers"]):
                self.convOutShape = shapeAfterConv(self.convOutShape,self.settings["convLayers"][i])
                self.convOutShape = (self.convOutShape[0],self.convOutShape[1]//2, self.convOutShape[2]//2)

            self.convOutLength = self.convOutShape[0] * self.convOutShape[1] * self.convOutShape[2]
            linearLayerDims = [self.convOutLength, 100,100, outDim]

            if "linearLayerDims" in self.settings:
                linearLayerDims = [self.convOutLength] + self.settings["linearLayerDims"] + [outDim]

            self.convLayers = [nn.Conv2d(i,j,k) for i,j,k in self.settings["convLayers"]]
            self.linearLayers = [nn.Linear(linearLayerDims[i],linearLayerDims[i+1]) for i in range(len(linearLayerDims)-1)]
            self.inDim = inDim

            self.setParameters()

        def setParameters(self):
            # pass
            tmp = []
            for layer in self.convLayers:
                tmp.extend(layer.parameters())
            for layer in self.linearLayers:
                tmp.extend(layer.parameters())
            self.myParameters = nn.ParameterList(tmp)

        def forward(self, x):
            x = torch.tensor(x, dtype = torch.float).cuda()
            x = x.view(-1,*self.inDim) # should be the shape of images coming in.

            for convLayer in self.convLayers:
                x = nn.functional.relu(convLayer(x))
                x = nn.MaxPool2d((2,2))(x)

            x = x.view(-1,self.convOutLength)

            for linearLayer in self.linearLayers[:-1]:
                x = nn.functional.relu(linearLayer(x))

            if self.settings["task"] == "classification":
                smOp = nn.Softmax(dim=1)
                return smOp(self.linearLayers[-1](x))
            elif self.settings["task"] == "regression":
                return self.linearLayers[-1](x)
            else:
                raise NotImplementedError

    def __init__(self,inDim,outDim,debug=True,**_settings):
        self.settings = {
            "convInit": "he",
            "linearInit": "he",
            "task": "classification"
        }
        self.settings.update(_settings)

        self.debug = debug

        self.inDim = inDim
        self.outDim = outDim

        self.net = self.Model(inDim,outDim, **_settings).cuda() # make model.
        self.initializeWeights(
            convInit = self.settings["convInit"],
            linearInit = self.settings["linearInit"]
        )

        self.batchSize = 32
        self.learningRate = 1e-4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learningRate)

        # array to hold loss values over time for graphing.
        self.losses = {"training":[], "validation": []}
        self.accuracies = {"training":[], "validation": []}


    def initializeWeights(self, convInit = "he", linearInit = "he"):
        self.convInit = convInit
        self.linearInit = linearInit
        convInit = getRandomWeightInit(layerType = "conv", init = convInit)
        linearInit = getRandomWeightInit(layerType = "linear", init = linearInit)

        for layer in self.net.convLayers:
            shape = layer.weight.shape
            weight = convInit(shape)
            layer.weight.data = torch.from_numpy(weight).float().cuda()
            layer.bias.data = torch.zeros(layer.bias.shape).cuda()


        for layer in self.net.linearLayers:
            shape = layer.weight.shape
            weight = linearInit(shape)
            layer.weight.data = torch.from_numpy(weight).float().cuda()
            layer.bias.data = torch.zeros(layer.bias.shape).cuda()

    def fit(self, x, y, epochs = 10000, plotWhileTraining = True, valFrac = 0.1):
        nonce = np.random.random()
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
        self.valX = x[indicesComp].cuda()
        self.valY = y[indicesComp].cuda()

        lossF = torch.nn.MSELoss()

        batchesPerEpoch = trainN // self.batchSize
        if plotWhileTraining:
            fig,ax = plt.subplots()
            ax.autoscale(True)
            trainLine, = ax.plot(self.losses["training"], color='green')
            valLine, = ax.plot(self.losses["validation"], color='red')
            plt.ion()

        for e in range(epochs):
            for batch in range(batchesPerEpoch):
                indices = np.random.choice(trainN,self.batchSize)
                x = self.trainX[indices].cuda()
                y = self.trainY[indices].cuda()
                self.optimizer.zero_grad()
                trainLoss = lossF(self.net(x), y)
                trainLoss.backward()
                self.optimizer.step()

                if batch % 20 == 0:
                    trainLossScalar = float(trainLoss.data.cpu().numpy())
                    self.losses["training"].append(trainLossScalar)

                    validationLoss = lossF(self.net(self.valX),self.valY)
                    validationLossScalar = float(validationLoss.data.cpu().numpy())
                    self.losses["validation"].append(validationLossScalar)

                    if self.settings["task"] == "classification":
                        out = self.net(x).cpu().detach().numpy()
                        y = y.cpu().detach().numpy()
                        self.accuracies["training"].append(accuracy(out,y))

                        out = self.net(self.valX).cpu().detach().numpy()
                        y = self.valY.cpu().detach().numpy()
                        self.accuracies["validation"].append(accuracy(out,y))

                    if plotWhileTraining:
                        limit = 100
                        n = len(self.accuracies["validation"])
                        trainLine.set_data(np.arange(n),self.accuracies["training"])
                        valLine.set_data(np.arange(n),self.accuracies["validation"])
                        if e == epochs-1:
                            ax.set_xlim(0,n+1)
                            ax.set_ylim(0,max(self.accuracies["validation"])*1.4)
                        else:
                            ax.set_xlim(max(0,n-limit),n+1)
                            ax.set_ylim(0,max(self.accuracies["validation"][-limit:])*1.4)
                        plt.show()
                        plt.pause(0.0001)

            if self.debug and e % 20 == 0:
                print("Training Loss for epoch {}: {}".format(e, self.losses["training"][-1]))
                print("Validation Loss for epoch {}: {}".format(e, self.losses["validation"][-1]))


    def predict(self,x):
        pred = self.net(x)
        return pred.cpu().detach().numpy().copy()
