import numpy as np
from scipy.linalg import hadamard
import code


def isOrthogonal(M):
    length = len(M)
    for i in range(length):
        for j in range(i+1,length):
            if np.dot(M[i],M[j]) > 1e-6:
                return False
    return True

def isOrthoNormal(M):
    length = len(M)
    for i in range(length):
        if abs(np.sqrt(np.dot(M[i],M[i])) - 2) > 1e-6:
            return False
        for j in range(i+1,length):
            if np.dot(M[i],M[j]) > 1e-6:
                return False
    return True

def avgDistBetweenRows(M):
    dist = lambda row: np.sqrt(np.dot(row,row))
    distances = []
    l = len(M)
    for i in range(l):
        for j in range(i+1,l):
            distances.append(dist(M[i]-M[j]))
    return np.mean(distances)


def fixShape(weight,shape):
    """takes a weight matrix and returns a new weight matrix of shape 'shape'.
    It will choose a subset of the rows if shape has fewer rows than the weight matrix
    and will add He initialized rows if the shape has more rows than the weight matrix.
    We do this to make sure that the weight matrix has as many weight vectors as outputs"""

    weightDim = len(weight)
    if weightDim > shape[0]: # more inputs than outputs
        rowsToKeep = np.random.choice(weightDim, shape[0], replace=False)
        weight = weight[rowsToKeep]
    elif weightDim < shape[0]: # more outputs than inputs
        augShape = (shape[0] - weightDim, shape[1])
        augment = np.random.normal(0,np.sqrt(2)/np.sqrt(shape[1]), augShape)
        weight = np.concatenate((weight, augment), axis=0)

    return weight

def getRandomOrthoNormalBasis(n):
    Q,R = np.linalg.qr(np.random.randn(n,n))
    # I want the length of all rows to be sqrt(2) so that the components have variance 2/n
    for i in range(n):
        Q[i] *= np.sqrt(2)
    return Q

def getRandomOrthoOrdentBasis(n):
    Q = np.random.uniform(0,2*0.797*np.sqrt(2/n),(n,n))
    p = int(2**np.ceil(np.log2(n)))
    H = hadamard(p)
    selectedRowsOfH = np.random.choice(p,n,replace=False)
    H = H[selectedRowsOfH][:,:n]

    for i in range(n):
        Q[i] *= H[i]
    return Q



def getRandomOrthoNormalWeightMatrix(shape):
    weight = getRandomOrthoNormalBasis(shape[1])
    return fixShape(weight, shape)

def getRandomOrthogonalWeightMatrix(shape):
    weight = getRandomOrthoNormalBasis(shape[1])
    for i in range(shape[1]):
        weight[i] *= np.random.uniform(0.5,1.5,1)
    return fixShape(weight, shape)

def getRandomHeWeightMatrix(shape):
    return np.random.normal(0,np.sqrt(2/shape[1]), shape)

def getRandomOrthoOrdentWeightMatrix(shape):
    weight = getRandomOrthoOrdentBasis(shape[1])
    return fixShape(weight,shape)

def getRandomQuadrantSubsetWeightMatrix(shape):

    def generateRandomQuadrant(n):
        quadrant = []
        for _ in range(n):
            quadrant.append(np.random.choice(2)*2-1)
        return tuple(quadrant)


    n = shape[1] # dimensionality
    numVecs = min(2**n,shape[0])

    quadrants = set()
    while len(quadrants) < numVecs:
        quadrants.add(generateRandomQuadrant(n))

    quadrants = np.array(list(quadrants))
    weight = quadrants * np.abs(np.random.normal(0,np.sqrt(2/n),(len(quadrants),n)))

    return fixShape(weight, shape)



def getRandomWeightInit(layerType, init):
    initFunctions = {
        "he": getRandomHeWeightMatrix,
        "orthogonal": getRandomOrthogonalWeightMatrix,
        "orthonormal": getRandomOrthoNormalWeightMatrix,
        "orthoordent": getRandomOrthoOrdentWeightMatrix,
        "quadrantsubset": getRandomQuadrantSubsetWeightMatrix
    }

    if init not in initFunctions:
        print("initialization \"{}\" not supported!".format(init))
        return False

    if layerType == "conv":
        linearInitFunc = initFunctions[init]

        # def convInitFunction(shape):
        #     """convolutional weight shape is [out_channels, in_channels, numRows, numColumns]"""
        #     outChannels = []
        #     for i in range(shape[0]):
        #         filt = []
        #         for j in range(shape[1]):
        #             filt.append(linearInitFunc(shape[-2:]))
        #         outChannels.append(filt)
        #     return np.array(outChannels)

        def convInitFunction(shape):
            """convolutional weight whape is [outChannels, inChannels, numRows, numColumns]"""
            outChannels, inChannels, numRows, numColumns = shape
            weight = linearInitFunc(shape = (outChannels, inChannels * numRows * numColumns))
            return weight.reshape(shape)

        return convInitFunction

    elif layerType == "linear":
        return initFunctions[init]

    else:
        print("LayerType \"{}\" not Supported!".format(layerType))
        return False
