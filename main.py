import numpy as np
import sklearn
from sklearn import preprocessing, svm
import sklearn.metrics as sm

train_images = np.loadtxt('C:\\Users\\Student\\PycharmProjects\\SVM\\data\\train_images.txt')
train_labels = np.loadtxt('C:\\Users\\Student\\PycharmProjects\\SVM\\data\\train_labels.txt')
test_images = np.loadtxt('C:\\Users\\Student\\PycharmProjects\\SVM\\data\\test_images.txt')
test_labels = np.loadtxt('C:\\Users\\Student\\PycharmProjects\\SVM\\data\\test_labels.txt')

def sum_classifiers(train_data, train_labels, test_data, c):
    model = svm.SVC(c, kernel='linear')
    model.fit(train_data, train_labels)
    predicted_train_labels = model.predict(train_data)
    predicted_test_labels = model.predict(test_data)
    return predicted_train_labels, predicted_test_labels


def normalize_data(train_data, test_data, type=None):
    if type is None:
        print('----No normalization-----')
        return train_data, test_data
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_test_data
    if type == "minmax":
        scaler = preprocessing.MinMaxScaler
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_test_data
    if type == 'l1':
        scaled_train_data /= np.sum(np.abs(train_data), keepdims=True, axis=1)
        scaled_test_data  /= np.sum(np.abs(test_data), keepdims=True, axis=1)
        return scaled_test_data, scaled_test_data
    if type == 'l2':
        scaled_test_data /= np.sum(np.sqrt(train_data**2), keepdims=True, axis=1)
        scaled_train_data /= np.sum(np.sqrt(test_data**2), keepdims=True, axis=1)
        return scaled_test_data, scaled_test_data



scaled_train_images, scaled_test_images = normalize_data(train_images, test_images, 'standard')
predicted_train, predicted_test = sum_classifiers(scaled_train_images, train_labels,scaled_test_images, 0.8)
train_acc = sm.accuracy_score(predicted_train, train_labels)
test_acc = sm.accuracy_score(predicted_test, test_labels)
print(train_acc, test_acc)
