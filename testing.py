from datasets import datasets, oneHotEncode
from convNet import NeuralNet
from sklearn.metrics import accuracy_score as accuracy

from tensorboardX import SummaryWriter

from collections import OrderedDict
import numpy as np
import tqdm
import code

import time

writer = SummaryWriter()


# inits = ["he"]
# inits = ["he", "orthoordent"]
# inits = ["quadrantsubset", "he"]
# inits = ["he","orthogonal","orthoordent"]
# inits = ["he", "orthogonal", "orthoordent", "quadrantsubset"]
inits = ["he","orthogonal","orthonormal","orthoordent","quadrantsubset"]
nets = []
trials = 50
valFrac = 0.05
defaultEpochs = 100 # ignored for most datasets which have their own value


# datasets = datasets[2:3]
datasets = datasets[3:4]
for i,db in enumerate(datasets):
    if "epochs" in db:
        epochs = db["epochs"]
    else:
        epochs = defaultEpochs

    nets.append({key:{"loss":[],"accuracy":[]} for key in inits})
    pbar = tqdm.tqdm(total = len(inits)*trials)
    pbar.set_description("Dataset {} ({}):".format(i,db["name"]))

    n = len(db["data"])
    if "realShape" in db:
        inDim = db["realShape"]
    else:
        inDim = len(db["data"][0])

    outDim = len(db["target"][0])

    for _ in range(trials):
        for init in inits:
            convLayers = [(1,5,(5,5)),(5,5,(3,3))] # (i,j,k) = (in_channels, out_channels, filter_size)
            linearLayers = [40]
            net = NeuralNet(
                inDim, outDim,
                debug = False,
                linearInit="he",
                convInit=init,
                convLayers = convLayers,
                linearLayers = linearLayers,
                task = db["task"],
                valFrac = valFrac
            )
            net.fit(db["data"],db["target"], epochs = epochs, plotWhileTraining = False)
            nets[i][init]["loss"].append(net.losses)
            nets[i][init]["accuracy"].append(net.accuracies)
            pbar.update(1)

    pbar.close()
    extractLoss = lambda x: np.min(x["validation"])
    extractAccuracy = lambda x: np.max(x["validation"])
    losses = {init:[extractLoss(vals) for vals in nets[i][init]["loss"]] for init in inits}
    accuracy = {init:[extractAccuracy(vals) for vals in nets[i][init]["accuracy"]] for init in inits}
    for init in inits:
        print("Losses for {}: {}".format(init, losses[init]))
        print("Mean Losses for {}: {}".format(init, np.mean(losses[init])))
        print("")

    print("#"*80)

for dataset,networks in enumerate(nets):
    meanTrainingLoss =       {init: np.mean([net["training"] for net in networks[init]["loss"]], axis=0) for init in networks}
    meanTrainingAccuracy =   {init: np.mean([net["training"] for net in networks[init]["accuracy"]], axis=0) for init in networks}
    meanValidationLoss =     {init: np.mean([net["validation"] for net in networks[init]["loss"]], axis=0) for init in networks}
    meanValidationAccuracy = {init: np.mean([net["validation"] for net in networks[init]["accuracy"]], axis=0) for init in networks}

    HeValidationAccuracy = np.array([net["validation"] for net in networks["he"]["accuracy"]])
    validationAccuracyAdvantage = {init: np.array([net["validation"] for net in networks[init]["accuracy"]]) - HeValidationAccuracy for init in networks}

    meanValidationAccuracyAdvantage = {init: np.mean(validationAccuracyAdvantage[init], axis=0) for init in networks}
    stdDevValidationAccuracyAdvantage = {init: np.std(validationAccuracyAdvantage[init], axis=0)/np.sqrt(trials) for init in networks}
    pValueValidationAccuracyAdvantage = {init: meanValidationAccuracyAdvantage[init]/stdDevValidationAccuracyAdvantage[init] for init in networks}

    for init in networks:
        for i,val in enumerate(meanTrainingLoss[init]):
            s = "{}_{}".format(dataset,"trainingLoss")
            writer.add_scalars(s,{init:val},i)

        for i,val in enumerate(meanValidationLoss[init]):
            s = "{}_{}".format(dataset,"validationLoss")
            writer.add_scalars(s,{init:val},i)

        for i,val in enumerate(meanTrainingAccuracy[init]):
            s = "{}_{}".format(dataset,"trainingAccuracy")
            writer.add_scalars(s,{init:val},i)

        for i,val in enumerate(meanValidationAccuracy[init]):
            s = "{}_{}".format(dataset,"validationAccuracy")
            writer.add_scalars(s,{init:val},i)

        for i,val in enumerate(meanValidationAccuracyAdvantage[init]):
            s = "{}_{}".format(dataset,"Mean of accuracy advantage over He")
            writer.add_scalars(s,{init:val},i)

        for i,val in enumerate(stdDevValidationAccuracyAdvantage[init]):
            s = "{}_{}".format(dataset,"Standard Deviation of accuracy advantage over He")
            writer.add_scalars(s,{init:val},i)

        for i,val in enumerate(pValueValidationAccuracyAdvantage[init]):
            s = "{}_{}".format(dataset,"p-value of accuracy advantage over He")
            writer.add_scalars(s,{init:val},i)



    # record the mean distance between all init accuracies and He accuracies
    # for init in networks:
    #     for i,he in enumerate(meanValidationAccuracy["he"]):
    #         val = (meanValidationAccuracy[init][i] - he)
    #         s = "{}_{}".format(dataset, "mean advantage over He")
    #         writer.add_scalars(s,{init:val}, i)

code.interact(local=locals())
