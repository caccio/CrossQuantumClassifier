import xquantum.blas as xqla
import xquantum.qdk as xqqdk
import xquantum.qiskit as xqqkit
from typing import NamedTuple, List, Mapping, Any
import numpy as np
from enum import Enum, IntEnum
import math, random
import pandas as pd

# Engine

class EngineType(Enum):
    BLAS = 0
    QDK = 1
    QISKIT = 2

    def __str__(self):
        return self.name

class Engine(NamedTuple):
    type: EngineType
    config: Mapping[str, Any]

    def __str__(self):
        return "{0} with config {1}".format(self.type, self.config)

# Network structure

class Pauli(IntEnum):
    X = 1
    Y = 2
    Z = 3

    def __repr__(self):
        return "{}".format(self.value)

class RotationBlock(NamedTuple):
    fixedQubitIndex: int
    rotationQubitIndex: int
    rotationParameterIndex: int
    rotationAxis: Pauli

    def __str__(self):
        return "RB({0},{1},{2},{3})".format(self.fixedQubitIndex, self.rotationQubitIndex, self.rotationParameterIndex,
                                            self.rotationAxis.name)

class CircuitModel(NamedTuple):
    network: List[RotationBlock]
    measureQubitIndex: int
    # measureParameterIndex : int
    biasParameterIndex: int

    def __str__(self):
        return "\tNetwork: {0}\n\tmeasureQubit: {1}\n\tbiasParameter: {2}".format(
            ['{0}'.format(r) for r in self.network], self.measureQubitIndex, self.biasParameterIndex)

# Hyper Params and Outcome

class HyperParams(NamedTuple):
    epochs: int
    batchSize: int
    shots: int
    learningRate: float
    decay: float
    momentum: float
    padding: bool
    pad: float

    def __str__(self):
        return "HyperParams:\n\tepochs: {0}\n\tbatchSize: {1}\n\tshots: {2}\n\tlearningRate: {3:.3f}\n\tdecay: {4}\n\tmomentum: {5:.3f}\n\tpad: {6:.3f}".format(
            self.epochs, self.batchSize, self.shots, self.learningRate, self.decay, self.momentum,
            self.pad if self.padding else 0)

class Outcome(NamedTuple):
    label: int
    probability: float
    pi: float

    def __str__(self):
        return "Label: {0}  Pr({0}): {1:.5f}  pi: {2:.5f}".format(self.label, self.probability, self.pi)

# Functions

def trainModel(engine, circuitModel, hyperParams, params, trainingFeatureSet, labelSet):
    learningHistory = pd.DataFrame([], columns=['Epoch', 'Error', 'Learning rate', 'Params'])
    if len(trainingFeatureSet) != len(labelSet):
        print("Training and Label set do not have the same length!")
        return (params, 1, learningHistory)
    normalizedFeatureSet = _normalizeFeatureSet(trainingFeatureSet, hyperParams,
                                                _circuitModelRegisterSize(circuitModel))
    bestEpoch = 0
    learningRate = hyperParams.learningRate
    bestParams = params
    bestError = _errorEstimation(hyperParams, labelSet,
                                 _batchClassify(engine, circuitModel, hyperParams, params, normalizedFeatureSet))
    learningHistory = learningHistory.append(
        {
            'Epoch': 0,
            'Error': bestError,
            'Learning rate': learningRate,
            'Params': params
        }, ignore_index=True)
    batches = math.ceil(len(normalizedFeatureSet) / hyperParams.batchSize)
    indexes = list(range(len(normalizedFeatureSet)))
    epochParams = bestParams
    epochError = bestError
    batchParams = epochParams
    batchCorrections = [0] * len(batchParams)
    for r in range(hyperParams.epochs):
        random.shuffle(indexes)
        trainingFeatureSet = list(map(lambda i: normalizedFeatureSet[i], indexes))
        trainingLabelSet = list(map(lambda i: labelSet[i], indexes))
        print(
            "Starting epoch {0}/{1} - last epoch error {3:.3f} - best error {4:.3f} in epoch {5} - learningRate {2:.3f}"
            .format(r + 1, hyperParams.epochs, learningRate, epochError, bestError, bestEpoch))
        for b in range(batches):
            print("\tTraining model - batch {0}/{1} - params {2}".format(b + 1, batches,
                                                                         [np.around(x, 3) for x in batchParams]))
            batchStart = b * hyperParams.batchSize
            batchEnd = min((b + 1) * hyperParams.batchSize, len(normalizedFeatureSet))
            (batchParams, batchCorrections) = _paramsOptimizaiton(engine, circuitModel, hyperParams, learningRate,
                                                                  batchParams, batchCorrections,
                                                                  trainingFeatureSet[batchStart:batchEnd],
                                                                  trainingLabelSet[batchStart:batchEnd])
            # batchError = _errorEstimation(hyperParams, labelSet, _batchClassify(engine, circuitModel, hyperParams, batchParams, normalizedFeatureSet))
        epochParams = batchParams
        epochError = _errorEstimation(
            hyperParams, labelSet, _batchClassify(engine, circuitModel, hyperParams, epochParams, normalizedFeatureSet))
        learningHistory = learningHistory.append(
            {
                'Epoch': (r + 1),
                'Error': epochError,
                'Learning rate': learningRate,
                'Params': batchParams
            },
            ignore_index=True)
        learningRate = _annealing(hyperParams, learningRate, r, (bestError - epochError))
        if epochError < bestError:
            bestError = epochError
            bestParams = epochParams
            bestEpoch = r + 1
            print("\t** New best epoch - error {0:.3f} - params {1}".format(bestError,
                                                                            [np.around(x, 3) for x in bestParams]))
    return (bestParams, bestError, learningHistory)

def _paramsOptimizaiton(engine, circuitModel, hyperParams, learningRate, params, corrections, normalizedFeatureSet,
                        labelSet):
    # calculate gradient of error function by params
    outcomes = _batchClassify(engine, circuitModel, hyperParams, params, normalizedFeatureSet)
    mse = [2 * (p - l) for p, l in zip(map(lambda o: o.pi, outcomes), labelSet)]  # MSE derivative coefficients
    gradient = [0] * len(params)
    for i in range(len(normalizedFeatureSet)):
        derr = _circuitDerivativeByParams(engine, circuitModel, hyperParams, params, normalizedFeatureSet[i])
        for p in range(len(params)):
            gradient[p] += mse[i] * derr[p] / len(normalizedFeatureSet)
    # update params
    newCorrections = [
        hyperParams.momentum * corrections[i] + (1 - hyperParams.momentum) * learningRate * gradient[i]
        for i in range(len(params))
    ]
    newParams = [params[i] - newCorrections[i] for i in range(len(params))]
    return (newParams, newCorrections)

def _annealing(hyperParams, learningRate, epoch, errorTrend):
    return hyperParams.learningRate / (1 + hyperParams.decay * epoch)

def _errorEstimation(hyperParams, labels, outcomes):
    return sum([(p - l)**2 for l, p in zip(labels, map(lambda o: o.label, outcomes))]) / len(labels)

def testModel(engine, circuitModel, hyperParams, params, testFeatureSet):
    normalizedFeatureSet = _normalizeFeatureSet(testFeatureSet, hyperParams, _circuitModelRegisterSize(circuitModel))
    outcomes = []
    batches = math.ceil(len(normalizedFeatureSet) / hyperParams.batchSize)
    for b in range(batches):
        batchStart = b * hyperParams.batchSize
        batchEnd = min((b + 1) * hyperParams.batchSize, len(normalizedFeatureSet))
        print("\tTesting model - batch {0}/{1}".format(b + 1, batches))
        outcomes = outcomes + _testModel(engine, circuitModel, hyperParams, params,
                                         normalizedFeatureSet[batchStart:batchEnd])
    return outcomes

def classify(engine, circuitModel, hyperParams, params, features):
    normalizedFeatures = _normalizeFeatures(features, hyperParams, _circuitModelRegisterSize(circuitModel))
    return _classify(engine, circuitModel, hyperParams, params, normalizedFeatures)

def _batchClassify(engine, circuitModel, hyperParams, params, normalizedFeatureSet):
    return _testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet)

# Common functions

def _circuitModelRegisterSize(circuit):
    size = circuit.measureQubitIndex
    for un in circuit.network:
        size = max(size, un.fixedQubitIndex, un.rotationQubitIndex)
    return size + 1

def _normalizeFeatures(features, hyperParams, circuitSize):
    f = len(features) + (1 if hyperParams.padding else 0)
    n = max((f - 1).bit_length(), circuitSize)
    values = [0] * (2**n)
    norm = 0
    for i in range(0, f):
        values[i] = features[i] if i < len(features) else hyperParams.pad if hyperParams.padding else 0
        norm += values[i]**2
    norm = math.sqrt(norm)
    return list(map(lambda x: x / norm, values))

def _normalizeFeatureSet(featureSet, hyperParams, circuitSize):
    return list(map(lambda f: _normalizeFeatures(f, hyperParams, circuitSize), featureSet))

def _defineOutcome(pr, pi):
    return Outcome(1 if pi > 0.5 else 0, pr, pi)

# Redirect to specific Engine methods

def _testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet):
    if engine.type == EngineType.BLAS:
        return [
            _defineOutcome(pr, pi)
            for (pr, pi) in xqla.testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet)
        ]
    if engine.type == EngineType.QDK:
        return [
            _defineOutcome(pr, pi)
            for (pr, pi) in xqqdk.testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet)
        ]
    if engine.type == EngineType.QISKIT:
        return [
            _defineOutcome(pr, pi)
            for (pr, pi) in xqqkit.testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet)
        ]

def _classify(engine, circuitModel, hyperParams, params, normalizedFeatures):
    (pr, pi) = (0, 0)
    if engine.type == EngineType.BLAS:
        (pr, pi) = xqla.classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures)
    if engine.type == EngineType.QDK:
        (pr, pi) = xqqdk.classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures)
    if engine.type == EngineType.QISKIT:
        (pr, pi) = xqqkit.classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures)
    return _defineOutcome(pr, pi)

def _circuitDerivativeByParams(engine, circuitModel, hyperParams, params, normalizedFeatures):
    if engine.type == EngineType.BLAS:
        return xqla.circuitDerivativeByParams(engine, circuitModel, hyperParams, params, normalizedFeatures)
    if engine.type == EngineType.QDK:
        return xqqdk.circuitDerivativeByParams(engine, circuitModel, hyperParams, params, normalizedFeatures)
    if engine.type == EngineType.QISKIT:
        return xqqkit.circuitDerivativeByParams(engine, circuitModel, hyperParams, params, normalizedFeatures)
