import xquantum as xq
import qsharp
import math

with open('{0}/qdk/qdk.qs'.format(xq.__path__[0]), 'r') as file:
    qdk = qsharp.compile(''.join(file.readlines()[1:-1]))

CircuitParameterDerivative = qdk[4]
Classify = qdk[5]

# from Cazzella.Quantum.CrossQuantum import Classify, CircuitParameterDerivative

def testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet):
    rotationSet = _encodeFeatureRotationSet(normalizedFeatureSet)
    return [_classifiedLableProbability(engine, circuitModel, hyperParams, params, r) for r in rotationSet]

def classifiedLableProbability(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    rotations = _encodeFeatureRotations(normalizedFeatures)
    return _classifiedLableProbability(engine, circuitModel, hyperParams, parameters, rotations)

def circuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    rotations = _encodeFeatureRotations(normalizedFeatures)
    return _circuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, rotations)

def _classifiedLableProbability(engine, circuitModel, hyperParams, parameters, rotations):
    p0 = 0.0
    for r in range(hyperParams.shots):
        l = _classify(circuitModel, parameters, rotations)
        p0 += 1 if l == 0 else 0
    p0 /= hyperParams.shots
    pi = p0 + parameters[circuitModel.biasParameterIndex]
    return (p0, pi)

def _classify(circuitModel, params, rotations):
    return Classify.simulate(circuit=circuitModel, parameters=params, rotations=rotations)

def _circuitDerivativeByParams(engine, circuitModel, hyperParams, params, rotations):
    dpar = [0] * len(params)
    for i in range(len(params)):
        if i == circuitModel.biasParameterIndex:
            dpar[i] = 1
        else:
            p0 = 0
            for r in range(hyperParams.shots):
                p0 += 1 - CircuitParameterDerivative.simulate(
                    circuit=circuitModel, parameters=params, rotations=rotations, parameterIndex=i)
            dpar[i] = (p0 / hyperParams.shots) - 0.5  # (2*p0-1)*0.5
    return dpar

def _encodeFeatureRotations(normalizedFeatures):
    n = (len(normalizedFeatures) - 1).bit_length()
    rotations = []
    for s in range(1, n + 1):
        ry = []
        for j in range(1, 2**(n - s) + 1):
            ny = sum(map(lambda l: normalizedFeatures[(2 * j - 1) * 2**(s - 1) + l - 1]**2, range(1, 2**(s - 1) + 1)))
            dy = sum(map(lambda l: normalizedFeatures[(j - 1) * 2**s + l - 1]**2, range(1, 2**s + 1)))
            ry.append(2 * math.asin(math.sqrt(ny / dy)) if ny > 0 else 0)
        rotations.append(ry)
    return rotations

def _encodeFeatureRotationSet(normalizedFeatureSet):
    return [_encodeFeatureRotations(nf) for nf in normalizedFeatureSet]
