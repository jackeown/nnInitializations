from MLPRegressor import MLPRegressor
from sklearn.datasets import load_breast_cancer, load_digits
import numpy as np



data = load_digits()
xs = data["data"]
ys = data["target"]
ys = ys.reshape((len(ys),1))


hparams = {
	"silent":False,
	"earlyStopping":False,
	"numEpochs":10000,
	"batchSize":32,
	"learningRate":1e-3,
	"numInputs":len(xs[0]),
	"numOutputs":1,
	"layers":[100,100],
	"dropoutProbability":1.0,
	"validationPercent":0.1
}


mlp = MLPRegressor(hparams)
mlp.fit(xs,ys)
