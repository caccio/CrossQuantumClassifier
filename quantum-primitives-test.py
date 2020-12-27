import numpy as np
import json
import math
import xquantum.xquantum as xq

with open('credentials.json') as f:
    credentials = json.load(f)

### Model definition (network and hyper-aprameter)

circuitModel = xq.CircuitModel([xq.RotationBlock(1, 0, 0, xq.Pauli.Y), xq.RotationBlock(0, 1, 1, xq.Pauli.Y)], 0, 2)

hyperParams = xq.HyperParams(1, 1, 8000, 0, 0, 0, False, 0.0)

### Engine configuration

engines = [
    xq.Engine(xq.EngineType.BLAS, {'derivativeMethod': 1}),
    xq.Engine(xq.EngineType.QDK, {}),
    xq.Engine(xq.EngineType.QISKIT, {}),
    xq.Engine(
        xq.EngineType.QISKIT, {
            'account': credentials['IBM-account'],
            'providerGroup': 'open',
            'providerProject': 'main',
            'providerBackend': 'ibmq_santiago'
        })
]

def errorRange(p, shots, conf=95):
    z = 1.96 if conf == 95 else 2.33 if conf == 98 else 2.58
    err = z * math.sqrt(p * (1 - p) / shots)
    return [p - err, p + err]

p = [np.pi / 3, np.pi / 4, 0]
x = [np.sqrt(.1), np.sqrt(.2), np.sqrt(.3), np.sqrt(.4)]
nf = xq._normalizeFeatures(x, hyperParams, xq._circuitModelRegisterSize(circuitModel))

expPi = 0.33054  # Maxima = 0.3305365232004617
expGrad = [-0.05289, 0.44620, 1]  # Maxima = [-0.05289080966970545, 0.4461961755974342, 1]

print("Classification test")
print("Expected result {0}".format(expPi))
for e in engines:
    o = xq._classify(e, circuitModel, hyperParams, p, nf)
    print("Engine {0}\n\tOutcome {1} - error range {2}".format(
        e, o, [np.around(x, 5) for x in errorRange(o.pi, hyperParams.shots)]))

print("\nPartial derivatives test")
print("Expected result {0}".format(expGrad))
for e in engines:
    q = xq._circuitDerivativeByParams(e, circuitModel, hyperParams, p, nf)
    print("Engine {0}\n\tPartial derivatives {1}".format(e, [np.around(x, 5) for x in q]))
