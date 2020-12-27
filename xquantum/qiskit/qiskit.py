import xquantum as xq
import math
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumCircuit, execute, Aer, IBMQ

def getQiskitBackend(engine):
    if 'providerGroup' in engine.config.keys():
        if engine.config['providerGroup'] == 'open' and engine.config['providerProject'] == 'main':
            IBMQ.save_account(engine.config['account'], overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(group=engine.config['providerGroup'], project=engine.config['providerProject'])
            return provider.get_backend(engine.config['providerBackend'])
    else:
        return Aer.get_backend('qasm_simulator')  # statevector_simulator   qasm_simulator

def testModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet):
    return _testParallelModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet)

def classifiedLableProbability(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    return _classifiedLableProbability(engine, circuitModel, hyperParams, parameters, normalizedFeatures)

def circuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    return _circuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, normalizedFeatures)

def _testParallelModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet):
    size = (len(normalizedFeatureSet[0]) - 1).bit_length()
    ecs = []
    circuitGate = _applyCircuit(size, circuitModel, params)
    for nf in normalizedFeatureSet:
        mc = QuantumCircuit(size, 1)
        mc.initialize(nf, list(range(size)))
        mc.append(circuitGate, list(range(size)))
        mc.measure(circuitModel.measureQubitIndex, 0)
        ecs.append(mc)
    backend = getQiskitBackend(engine)
    job = execute(ecs, backend, shots=hyperParams.shots)
    counts = [job.result().get_counts(qc) for qc in ecs]
    p0s = [(c['0'] if '0' in c.keys() else 0) / hyperParams.shots for c in counts]
    return [[p0, (p0 + params[circuitModel.biasParameterIndex])] for p0 in p0s]

def _testIterativeModel(engine, circuitModel, hyperParams, params, normalizedFeatureSet):
    return [_classifiedLableProbability(engine, circuitModel, hyperParams, params, nf) for nf in normalizedFeatureSet]

def _classifiedLableProbability(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    size = (len(normalizedFeatures) - 1).bit_length()
    qc = QuantumCircuit(size, 1)
    qc.initialize(normalizedFeatures, list(range(size)))
    qc.append(_applyCircuit(size, circuitModel, parameters), list(range(size)))
    qc.measure(circuitModel.measureQubitIndex, 0)
    backend = getQiskitBackend(engine)
    job = execute(qc, backend, shots=hyperParams.shots)
    count = job.result().get_counts(qc)
    p0 = (count['0'] if '0' in count.keys() else 0) / hyperParams.shots
    pi = p0 + parameters[circuitModel.biasParameterIndex]
    return (p0, pi)

# https://algassert.com/quirk#circuit={%22cols%22:[[1,{%22id%22:%22Ryft%22,%22arg%22:%221.982313%22}],[],[{%22id%22:%22Ryft%22,%22arg%22:%221.714144%22},%22%E2%80%A2%22],[],[{%22id%22:%22Ryft%22,%22arg%22:%221.910633%22},%22%E2%97%A6%22],[],[%22Amps2%22],[],[1,1,%22H%22],[%22%E2%80%A2%22,%22X%22,%22%E2%80%A2%22],[{%22id%22:%22Ryft%22,%22arg%22:%22pi/2%22},1,%22%E2%80%A2%22],[%22%E2%80%A2%22,%22X%22,%22%E2%80%A2%22],[%22X%22,%22%E2%80%A2%22,%22%E2%80%A2%22],[1,{%22id%22:%22Ryft%22,%22arg%22:%22pi/3%22},%22%E2%80%A2%22],[%22X%22,%22%E2%80%A2%22,%22%E2%80%A2%22],[{%22id%22:%22Ryft%22,%22arg%22:%22pi/4%22},1,%22%E2%80%A2%22],[%22Z%22,1,%22%E2%80%A2%22],[1,1,%22X%22],[%22%E2%80%A2%22,%22X%22,%22%E2%80%A2%22],[{%22id%22:%22Ryft%22,%22arg%22:%22pi/2+pi%22},1,%22%E2%80%A2%22],[],[%22%E2%80%A2%22,%22X%22,%22%E2%80%A2%22],[%22X%22,%22%E2%80%A2%22,%22%E2%80%A2%22],[1,{%22id%22:%22Ryft%22,%22arg%22:%22pi/3%22},%22%E2%80%A2%22],[%22X%22,%22%E2%80%A2%22,%22%E2%80%A2%22],[{%22id%22:%22Ryft%22,%22arg%22:%22pi/4%22},1,%22%E2%80%A2%22],[1,1,%22X%22],[1,1,%22H%22],[%22Density%22,%22Density%22,%22Density%22]],%22init%22:[0,0,1]}
def _circuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    return _iterativeCircuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, normalizedFeatures)

def _iterativeCircuitDerivativeByParams(engine, circuitModel, hyperParams, parameters, normalizedFeatures):
    size = (len(normalizedFeatures) - 1).bit_length()
    dpar = [0] * len(parameters)
    circuitGate = _applyCircuit(size, circuitModel, parameters).control(1)
    for i in range(len(parameters)):
        if i == circuitModel.biasParameterIndex:
            dpar[i] = 1
        else:
            shiftedParams = list(
                map(lambda j: parameters[j] if j != i else parameters[j] + math.pi, range(len(parameters))))
            shiftedGate = _applyCircuit(size, circuitModel, shiftedParams).control(1)
            # the size-th qubit is the ancilla
            qc = QuantumCircuit(size + 1, 1)
            qc.initialize(normalizedFeatures, list(range(size)))
            qc.h(size)
            qc.append(circuitGate, [size] + list(range(size)))
            qc.cz(size, circuitModel.measureQubitIndex)
            qc.x(size)
            qc.append(shiftedGate, [size] + list(range(size)))
            qc.x(size)
            qc.h(size)
            qc.measure(size, 0)
            backend = getQiskitBackend(engine)
            job = execute(qc, backend, shots=hyperParams.shots)
            count = job.result().get_counts(qc)
            p0 = (count['0'] if '0' in count.keys() else 0)
            dpar[i] = (p0 / hyperParams.shots) - 0.5  # (2*p0-1)*0.5
    return dpar

# https://algassert.com/quirk#circuit={%22cols%22:[[1,{%22id%22:%22Ryft%22,%22arg%22:%221.982313%22}],[],[{%22id%22:%22Ryft%22,%22arg%22:%221.714144%22},%22%E2%80%A2%22],[],[{%22id%22:%22Ryft%22,%22arg%22:%221.910633%22},%22%E2%97%A6%22],[],[%22Amps2%22],[],[%22%E2%80%A2%22,%22X%22],[{%22id%22:%22Ryft%22,%22arg%22:%22pi/2%22}],[%22%E2%80%A2%22,%22X%22],[%22X%22,%22%E2%80%A2%22],[1,{%22id%22:%22Ryft%22,%22arg%22:%22pi/3%22}],[%22X%22,%22%E2%80%A2%22],[{%22id%22:%22Ryft%22,%22arg%22:%22pi/4%22}],[%22Amps2%22]]}
def _applyCircuit(n, circuitModel, parameters):
    qc = QuantumCircuit(n)
    for un in circuitModel.network:
        qc.cx(un.rotationQubitIndex, un.fixedQubitIndex)
        if un.rotationAxis == xq.Pauli.X: qc.rx(parameters[un.rotationParameterIndex], un.rotationQubitIndex)
        if un.rotationAxis == xq.Pauli.Y: qc.ry(parameters[un.rotationParameterIndex], un.rotationQubitIndex)
        if un.rotationAxis == xq.Pauli.Z: qc.rz(parameters[un.rotationParameterIndex], un.rotationQubitIndex)
        qc.cx(un.rotationQubitIndex, un.fixedQubitIndex)
    # qc.ry(parameters[circuitModel.measureParameterIndex], circuitModel.measureQubitIndex)
    return qc.to_gate()
