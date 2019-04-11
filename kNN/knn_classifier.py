import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from math import sqrt


# generates normally distributed random variables
def generate_data(samples, n_classes):
    X, y = make_blobs(n_samples=samples, centers=n_classes, n_features=2, random_state=0)
    return X, y

def plot_data(X, y):
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red',1:'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color = colors[key])
    plt.show()

# kNN classifier
def classifier(X_train, y_train, X_test, k):
    y_pred = []
    for point in X_test:
        dist = [np.linalg.norm(x-point) for x in X_train]
        dist = list(zip(dist, y_train))
        dist.sort(key=lambda x : x[0])
        k_nn = dist[0:k]
        labels = [i[1] for i in k_nn]
        category = max(set(labels), key=labels.count)
        y_pred.append(category)
    return y_pred



if __name__ == "__main__":

    #create two distributions
    N = 1000
    centers = [(1, 1), (4, 4)]
    data, cat = generate_data(N,centers)

    #plot two distributions
    plot_data(data,cat)

    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, cat, test_size=0.33, random_state=0)


    k_range = [1,5,10,20,50,100]
    all_accuracy = []
    all_rmse = []

    best_accuracy = 0
    best_k = 0
    for k in k_range:
        y_pred = classifier(X_train, y_train, X_test, k)
        accuracy = accuracy_score(y_test, y_pred)
        rmse_val = sqrt(mean_squared_error(y_test, y_pred))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
        print('Accuracy score: %f' % accuracy)
        all_accuracy.append(accuracy)
        all_rmse.append(rmse_val)

    plt.figure()
    plt.plot(k_range, all_accuracy, '*')
    plt.xlabel('k')
    plt.ylabel('Accuracy score')
    plt.title('Accuracy score vs number of NN')

    plt.figure()
    plt.plot(k_range, all_rmse, '*')
    plt.xlabel('k')
    plt.ylabel('RMS error')
    plt.title('RMS error vs number of NN')

    print("Best accuracy score: %f" % best_accuracy)
    print("Best k-NN number: %d" % best_k)

    y_pred = classifier(X_train, y_train, X_test, best_k)

    plot_data(X_test,y_pred)
    plot_data(X_test,y_test)

