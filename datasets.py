import numpy as np
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import code


def oneHotEncode(labels,n):
    out = [[int(i==label) for i in range(0,n)] for label in labels]
    return np.array(out, dtype=np.float)


# This file contains datasets for testing.
# Each dataset is a dictionary with 3 keys: "data", and "targets", and "type" which
# should be either "regression" or "classification".
# "data" and "targets" will each map to numpy 2d ndarrays with one row being one datapoint.
# This standard way of representing a dataset will be beneficial for testing
# in a more standard way.

datasets = []

########################## Breast Cancer Dataset ###########################
db = load_breast_cancer()
db["target"] = db["target"].reshape((len(db["data"]),-1))

# Scale/Preprocess data
scaler = StandardScaler()
db["name"] = "Breast Cancer"
db["data"] = scaler.fit_transform(db["data"])
db["task"] = "classification"
db["oneHot"] = False
datasets.append(db)


########################## Boston Housing Dataset ##########################
db = load_boston()
db["target"] = db["target"].reshape((len(db["data"]),-1))

# Scale/Preprocess data
scaler = StandardScaler()
db["name"] = "Boston Housing"
db["data"] = scaler.fit_transform(db["data"])
db["task"] = "regression"
datasets.append(db)


########################## Not MNIST Digit Dataset #############################
db = load_digits()
db["target"] = db["target"].reshape((len(db["data"]),-1))

# Scale/Preprocess data
scaler = StandardScaler()
db["name"] = "Not MNIST Digits"
db["data"] = scaler.fit_transform(db["data"])
db["task"] = "classification"
db["oneHot"] = False
db["realShape"] = (1,8,8)
datasets.append(db)

############################ MNIST Digit Dataset ##############################

mnist = tf.keras.datasets.mnist
(tx,ty),(vx,vy) = mnist.load_data()
db = {"data": np.concatenate((tx,vx)),"target":np.concatenate((ty,vy)) }
scaler = StandardScaler()

db["name"] = "MNIST digits"
# code.interact(local=locals())
db["data"] = db["data"].reshape(-1,28*28)
db["data"] = scaler.fit_transform(db["data"])
# db["data"] = db["data"] / 255.0
db["target"] = oneHotEncode(db["target"],10)
db["task"] = "classification"
db["oneHot"] = True
db["realShape"] = (1,28,28)
db["epochs"] = 12
datasets.append(db)

############################ Diabetes Dataset ##############################
db = load_diabetes()
db["target"] = db["target"].reshape((len(db["data"]),-1))

# Scale/Preprocess data
scaler = StandardScaler()
db["name"] = "Diabetes"
db["data"] = scaler.fit_transform(db["data"])
db["task"] = "regression"
datasets.append(db)


############################ Iris Dataset ##################################
db = load_iris()
db["target"] = db["target"].reshape((len(db["data"]),-1))

# Scale/Preprocess data
scaler = StandardScaler()
db["name"] = "Iris Flowers"
db["data"] = scaler.fit_transform(db["data"])
db["task"] = "classification"
db["oneHot"] = False
datasets.append(db)
