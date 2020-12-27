import numpy as np

# Basic gates definition
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.sqrt(2) / 2 * np.array([[1, 1], [1, -1]])

def Rx(t):
    return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])

def Ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])

def Rz(t):
    return np.array([[np.e**(-1j * t / 2), 0], [0, np.e**(1j * t / 2)]])

# Return I(2^n)
def ID(n):
    return np.array([[1]]) if n == 0 else np.kron(ID(n - 1), I)

# Apply U on qubit t controlled by qubit c in a s qubits operator
def CNU(s, c, t, U):
    if c > t:
        return ID(s) + np.kron(np.kron(np.kron(np.kron(ID(s - c), ID(1) - Z), ID(c - t - 1)), U - ID(1)), ID(t - 1)) / 2 # yapf: disable
    else:
        return ID(s) + np.kron(np.kron(np.kron(np.kron(ID(s - t), U - ID(1)), ID(t - c - 1)), ID(1) - Z), ID(c - 1)) / 2 # yapf: disable

# Apply operator U to qubit t in a s qubit operator
def PU(s, t, U):
    return np.kron(np.kron(ID(s - t), U), ID(t - 1))

# Circuit Unitary Operator Ra in position r with control qubit f in a s qubits circuit
def RGP(s, f, r, a, p):
    U = I
    if a == 1:
        U = Rx(p)
    if a == 2:
        U = Ry(p)
    if a == 3:
        U = Rz(p)
    return CNU(s, r, f, X) @ PU(s, r, U) @ CNU(s, r, f, X)

# Aply the circuit C to the vector w and measure the probability that qubit q is 0 - Pr(q=0)
def M(s, w, q):
    return (np.transpose(np.conjugate(w)).dot(PU(s, q, Z)).dot(w)[0][0].real + 1) / 2

# Toffoli gate with a and b control qubit and t target qubit for a s sized registry
def T(n, a, b, x):
    m = 2**(n - a) + 2**(n - b)
    d = lambda i: 2**(n - x) if (2**(n - x)) & i == 0 else -(2**(n - x))
    T = np.array([([0] * 2**n)] * 2**n)
    for i in range(2**n):
        for j in range(2**n):
            T[i][j] = 1 if (i & m == m and j == d(i) + i) or (i & m != m and j == i) else 0
    return T

# Create a controlled version of RGP gate with an extra ancila qubit with index s
def CRGP(s, f, r, a, p):
    U = I
    if a == 1:
        U = Rx(p)
    if a == 2:
        U = Ry(p)
    if a == 3:
        U = Rz(p)
    return T(s, s, r, f) @ CNU(s, s, r, U) @ T(s, s, r, f)

def testModel(engine, circuitModel, hyperParams, params, normalizedFeaturesSet):
    # print("\t\t{0} with params {1}".format(engine, params))
    return [_classifiedLableProbability(engine, circuitModel, hyperParams, params, nf) for nf in normalizedFeaturesSet]

def classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures):
    return _classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures)

def circuitDerivativeByParams(engine, circuitModel, hyperParams, params, normalizedFeatures):
    if (engine.config['derivativeMethod'] == 0):
        return _finiteDifferenceDerivative(engine, circuitModel, hyperParams, params, normalizedFeatures)
    if (engine.config['derivativeMethod'] == 1):
        return _innerProductDerivative(engine, circuitModel, hyperParams, params, normalizedFeatures)

def _classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures):
    n = (len(normalizedFeatures) - 1).bit_length()
    p0 = M(n, _applyCircuit(engine, circuitModel, hyperParams, params, normalizedFeatures),
           circuitModel.measureQubitIndex + 1)
    return (p0, p0 + params[circuitModel.biasParameterIndex])

def _applyCircuit(engine, circuitModel, hyperParams, params, normalizedFeatures):
    n = (len(normalizedFeatures) - 1).bit_length()
    C = ID(n)
    for un in circuitModel.network:
        C = RGP(n, un.fixedQubitIndex + 1, un.rotationQubitIndex + 1, un.rotationAxis,
                params[un.rotationParameterIndex]).dot(C)
    # C = PU(n,circuitModel.measureQubitIndex+1,Ry(params[circuitModel.measureParameterIndex])).dot(C)
    return C.dot(np.transpose(np.array([normalizedFeatures])))

def _finiteDifferenceDerivative(engine, circuitModel, hyperParams, params, normalizedFeatures):
    step = np.pi / 100000
    d = [0] * len(params)
    for i in range(len(params)):
        if i == circuitModel.biasParameterIndex:
            d[i] = 1
        else:
            shiftedParams = list(map(lambda j: params[j] if j != i else params[j] + step, range(len(params))))
            x0 = _classifiedLableProbability(engine, circuitModel, hyperParams, params, normalizedFeatures)[1]
            x1 = _classifiedLableProbability(engine, circuitModel, hyperParams, shiftedParams, normalizedFeatures)[1]
            d[i] = (x1 - x0) / step + step**2
    return d

def _innerProductDerivative(engine, circuitModel, hyperParams, params, normalizedFeatures):
    n = (len(normalizedFeatures) - 1).bit_length()
    d = [0] * len(params)
    for i in range(len(params)):
        if i == circuitModel.biasParameterIndex:
            d[i] = 1
        else:
            shiftedParams = list(map(lambda j: params[j] if j != i else params[j] + np.pi, range(len(params))))
            w = _applyCircuit(engine, circuitModel, hyperParams, params, normalizedFeatures)
            ws = _applyCircuit(engine, circuitModel, hyperParams, shiftedParams, normalizedFeatures)
            d[i] = ((w.T.conjugate() @ PU(n, circuitModel.measureQubitIndex + 1, Z) @ ws)[0][0].real) / 2
    return d
