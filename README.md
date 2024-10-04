# Descriptor Effectiveness: Hu Moments and LBP

This Python project uses computer vision to classify images into two groups using a supervised model. The program's main focus is feature extraction using the Hu Moments and LBP descriptors. Initially, the program extracts these features, followed by model training. After training, the program performs tests using the MLP (Multilayer Perceptron), Random Forest (RF) and SVM (Support Vector Machine) classifiers. At the end of the tests, a report is generated containing a confusion matrix, providing a detailed analysis of the accuracy achieved in classifying the images. Lobboㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ ## Repository

## Classifier and Accuracy

The classification process in this project is driven by a supervised model using the MLP (Multi-Layer Perceptron), Support Vector Machine (SVM), and Random Forest (RF) classifiers. The classification result is presented in a confusion matrix.ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ

**RESULTS OBTAINED**
| Hu Moments ㅤ | LBP |
| ---------------------------- | ------------------------ |
| MLP = 50.00% Accuracy ㅤ | MLP = 89.29% Accuracyㅤ |
| SVM = 53.57% Accuracy ㅤ | SVM = 73.21% Accuracy ㅤ|
| RF = 60.71% Accuracy ㅤ | RF = 98.21% Accuracyㅤ |

## Installation

In a Linux environment, open the terminal and run the following commands:

```bash
# Install Python

sudo apt install python3
```

```bash
# Install the Python package manager (pip)

sudo apt install python3-pip
```

```bash
# Install the necessary libraries

pip install split-folders
pip install opencv-python
pip install numpy
pip install scikit-image
pip install scikit-learn
pip install progress
pip install matplotlib
```
