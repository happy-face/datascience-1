import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from scipy.stats import norm


def calculate_pdf(x, x_mean, x_std):
    pdf_val = norm(x_mean, x_std).pdf(x)
    return pdf_val

def calculate_probabilities(X,y):

    def class_probabilities(y):
        unique, counts = np.unique(y, return_counts=True)
        p_class = counts/len(y)
        return p_class

    def conditional_probabilities(X,y):
        df = pd.DataFrame(X)
        p_all = []
        x_mean_all = []
        x_std_all = []
        for i in np.unique(y):
            df_new = df[y == i]
            # calculate class conditional mean and std
            x_mean = df_new.mean(axis=0)
            x_std = df_new.std(axis=0)
            p_1 = []
            # estimate conditional probabilities for each attribute (1-4)
            for j in range(X.shape[1]):
                a = df_new[j]
                a_mean = x_mean[j]
                a_std = x_std[j]
                a_pdf = calculate_pdf(a, a_mean, a_std)
                p_1.append(a_pdf)
            p_all.append(p_1)
            x_mean_all.append(x_mean)
            x_std_all.append(x_std)

        #return p_all, x_mean_all, x_std_all
        return x_mean_all, x_std_all

    class_prob = class_probabilities(y)
    #conditional_prob, x_mean, x_std = conditional_probabilities(X,y)
    x_mean, x_std = conditional_probabilities(X, y)

    #return class_prob, conditional_prob, x_mean, x_std
    return class_prob, x_mean, x_std

def predict_class(X,y, x_mean, x_std, class_prob):
    pred_class = []
    for item in X:
        #for each class 0,1,2
        prob_class =[]
        for j in np.unique(y):
            class_mean = x_mean[j]
            class_std = x_std[j]
            prob_all = []
            for k in range(X.shape[1]):
                prob = calculate_pdf(item[k], class_mean[k], class_std[k])
                prob_all.append(prob)
            prob_total = np.prod(prob_all)*class_prob[j]
            prob_class.append(prob_total)
            class_val = prob_class.index(max(prob_class))
        pred_class.append(class_val)
    return pred_class


if __name__ == "__main__":

    df = pd.read_csv("Iris.csv")
    df.head()

    df['category_id'] = df['Species'].factorize()[0]
    category_id_df = df[['Species','category_id']].sort_values('category_id').drop_duplicates()
    category_to_id = dict(category_id_df.values)

    x_columns = df.columns[1:5]
    X = df[x_columns].values
    y = df['category_id'].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

    #p_class, p_conditional, X_mean, X_std = calculate_probabilities(X_train, y_train)
    p_class, X_mean, X_std = calculate_probabilities(X_train, y_train)

    y_pred = predict_class(X_test,y_train, X_mean, X_std, p_class)

    print(accuracy_score(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))


