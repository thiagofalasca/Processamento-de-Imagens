import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import time
from datetime import datetime


def main():
    mainStartTime = time.time()
    results = []
    modelNames = ['MLP', 'SVM', 'RF']

    classifiers = [
        ('MLP', train_MLP, predictMLP),
        ('SVM', train_SVM, predictSVM),
        ('RF', train_RandomForest, predictRandomForest)
    ]

    for model_name, train_func, predict_func in classifiers:
        print(f'[INFO] *********{model_name}**********.')
        accuracy = train_and_predict(model_name, train_func, predict_func)
        results.append(accuracy)

    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Total code execution time: {elapsedTime}s')
    plotResults(modelNames, results)
    print(f'[INFO] Done, access ./Projeto/results to see the results')


def train_and_predict(model_name, train_func, predict_func):
    mainStartTime = time.time()
    trainFeaturePath = './Projeto/features_labels/train/'
    testFeaturePath = './Projeto/features_labels/val/'
    featureFilename = 'features.csv'
    labelFilename = 'labels.csv'
    encoderFilename = 'encoder_classes.csv'

    print(f'[INFO] ========= TRAINING PHASE ========= ')
    trainFeatures = getFeatures(trainFeaturePath, featureFilename)
    trainEncodedLabels = getLabels(trainFeaturePath, labelFilename)
    model = train_func(trainFeatures, trainEncodedLabels)

    print(f'[INFO] =========== TEST PHASE =========== ')
    testFeatures = getFeatures(testFeaturePath, featureFilename)
    testEncodedLabels = getLabels(testFeaturePath, labelFilename)
    encoderClasses = getEncoderClasses(testFeaturePath, encoderFilename)
    predictedLabels = predict_func(model, testFeatures)

    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time for {model_name}: {elapsedTime}s')
    accuracy = plotConfusionMatrix(model_name, encoderClasses, testEncodedLabels, predictedLabels)
    return accuracy


def train_model(model, trainData, trainLabels, modelName):
    print(f'[INFO] Training the {modelName} model...')
    startTime = time.time()
    trained_model = model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return trained_model


def train_MLP(trainData, trainLabels):
    mlp_model = MLPClassifier(random_state=1, hidden_layer_sizes=(5000,), max_iter=1000)
    return train_model(mlp_model, trainData, trainLabels, "MLP")


def train_RandomForest(trainData, trainLabels):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    return train_model(rf_model, trainData, trainLabels, "Random Forest")


def train_SVM(trainData, trainLabels):
    svm_model = svm.SVC(kernel='linear', C=1, random_state=42)
    return train_model(svm_model, trainData, trainLabels, "SVM")


def predictMLP(mlp_model, testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = mlp_model.predict(testData)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels


def predictRandomForest(rf_model, testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = rf_model.predict(testData)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels


def predictSVM(svm_model, testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = svm_model.predict(testData)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels


def getFeatures(path, filename):
    features = np.loadtxt(path+filename, delimiter=',')
    return features


def getLabels(path, filename):
    encodedLabels = np.loadtxt(path+filename, delimiter=',', dtype=int)
    return encodedLabels


def getEncoderClasses(path, filename):
    encoderClasses = np.loadtxt(path+filename, delimiter=',', dtype=str)
    return encoderClasses


def getCurrentFileNameAndDateTime():
    fileName = os.path.basename(__file__).split('.')[0]
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName+dateTime


def plotConfusionMatrix(model_name, encoderClasses, testEncodedLabels, predictedLabels):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = encoderClasses
    test = encoder.inverse_transform(testEncodedLabels)
    pred = encoder.inverse_transform(predictedLabels)
    print(f'[INFO] Plotting confusion matrix and accuracy...')
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test, pred, ax=ax, colorbar=False, cmap=plt.cm.Greens)
    plt.suptitle('Confusion Matrix: ' + model_name + getCurrentFileNameAndDateTime(), fontsize=18)
    accuracy = metrics.accuracy_score(testEncodedLabels, predictedLabels)*100
    plt.title(f'Accuracy: {accuracy}%', fontsize=18, weight='bold')
    plt.savefig('./Projeto/results/' + model_name + getCurrentFileNameAndDateTime(), dpi=300)
    print(f'[INFO] Plotting done!')
    return accuracy


def plotResults(modelNames, results):
    fig, ax = plt.subplots()
    bar_container = ax.bar(modelNames, results, color=['red', 'green', 'blue', 'cyan'])
    ax.set_ylabel('Accuracy', weight='bold')
    ax.set_xlabel('Models', weight='bold')
    ax.set_title('Model comparison', fontsize=18, weight='bold')
    ax.bar_label(bar_container, fmt='{:,.2f}%')
    plt.savefig('./Projeto/results/'+getCurrentFileNameAndDateTime(), dpi=300)
    print(f'[INFO] Plotting final results done in ./Projeto/results/{getCurrentFileNameAndDateTime()}')


if __name__ == "__main__":
    main()
