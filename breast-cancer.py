import csv
import xquantum as xq
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

### Data Preparation

features = []
labels = []

# Dataset : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
with open('breast-cancer-wisconsin.csv') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    for row in rows:
        features.append([float(x) for x in row[1:10]])
        labels.append(int(row[10]) / 2 - 1)

trainingSetSize = int(0.70 * len(features))

sample = {
    'TrainingData': {
        'Features': [features[i] for i in range(trainingSetSize)],
        'Labels': [labels[i] for i in range(trainingSetSize)]
    },
    'ValidationData': {
        'Features': [features[i] for i in range(trainingSetSize, len(features))],
        'Labels': [labels[i] for i in range(trainingSetSize, len(features))]
    }
}

# scale features
scaler = MinMaxScaler().fit(sample['TrainingData']['Features'])
sample['TrainingData']['Features'] = scaler.transform(sample['TrainingData']['Features'])
sample['ValidationData']['Features'] = scaler.transform(sample['ValidationData']['Features'])

### Model definition (network and hyper-aprameter)

circuitModel = xq.CircuitModel([
    xq.RotationBlock(0, 1, 0, xq.Pauli.Y),
    xq.RotationBlock(1, 2, 1, xq.Pauli.Y),
    xq.RotationBlock(2, 3, 2, xq.Pauli.Y),
    xq.RotationBlock(3, 0, 3, xq.Pauli.Y),
    xq.RotationBlock(2, 0, 4, xq.Pauli.Y)
], 0, 5)

hyperParams = xq.HyperParams(epochs=20,
                             batchSize=20,
                             shots=200,
                             learningRate=0.157,
                             decay=0.02,
                             momentum=0.9,
                             padding=True,
                             pad=0.1)

### Engine configuration

engine = xq.Engine(xq.EngineType.QISKIT, {})

### Train the model

parameters = [0, 0, 0, 0, 0, 0]

print("Engine {0}".format(engine))
print("Circuit model {0}".format(circuitModel))
print("Start training with params {0} and {1}".format([np.around(x, 3) for x in parameters], hyperParams))
startingTime = time.time()
(fittedParams, error, learningHystory) = xq.trainModel(engine, circuitModel, hyperParams, parameters,
                                                       sample['TrainingData']['Features'],
                                                       sample['TrainingData']['Labels'])
print("Training ended in {2:.0f} seconds with error {0:.3f} and params {1} ".format(
    error, [np.around(x, 3) for x in fittedParams],
    time.time() - startingTime))

### Test the model

print("Test on training set")
startingTime = time.time()
actual_labels = sample['TrainingData']['Labels']
classified_labels = [
    o.label for o in xq.testModel(engine, circuitModel, hyperParams, fittedParams, sample['TrainingData']['Features'])
]
print("Test ended in {0:.0f} seconds with the following results".format(time.time() - startingTime))
print("Confusion Matrix")
print(confusion_matrix(actual_labels, classified_labels))
print("Classification Report")
print(classification_report(actual_labels, classified_labels))

print("Test on validation set")
startingTime = time.time()
actual_labels = sample['ValidationData']['Labels']
classified_labels = [
    o.label for o in xq.testModel(engine, circuitModel, hyperParams, fittedParams, sample['ValidationData']['Features'])
]
print("Test ended in {0:.0f} seconds with the following results".format(time.time() - startingTime))
print("Confusion Matrix")
print(confusion_matrix(actual_labels, classified_labels))
print("Classification Report")
print(classification_report(actual_labels, classified_labels))

### Plot the learning curve

plt.plot(learningHystory['Epoch'], learningHystory['Error'])
plt.xlabel('Epochs')
plt.ylabel('Error rate')
plt.title('Learning curve')
plt.grid(True)
plt.show()