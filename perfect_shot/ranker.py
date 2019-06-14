import itertools
import numpy as np
import argparse
from scipy import stats
import pylab as pl
import cv2
from sklearn import svm, linear_model
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd



# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV dataset file")
    parser.add_argument("-l", "--labels", required=True, help = "Input CSV labels file")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df = pd.read_csv(args.input, '\t')
    X = df[['im_file', 'blur', 'noise', 'brightness']]
    X['brightness'] = X['brightness'].str.strip('[]').astype(float)
    X = X.sort_values('im_file')
    X = X.drop(columns=['im_file'])
    X = X.reset_index(drop=True)

    y_df = pd.read_csv(args.labels, '\t')
    y_df = y_df[0:(X.shape[0])]
    y = y_df['Unnamed: 1']

    blocks = df['set']

    sss = StratifiedShuffleSplit(n_splits=2, test_size=.5)

    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        b_train, b_test = blocks.iloc[train_index], blocks.iloc[test_index]


    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    b_train = b_train.values
    b_test = b_test.values

    # plot the result
    '''idx = (b_train == 0)
    pl.scatter(X_train[idx, 0], X_train[idx, 1], c=y_train[idx],
        marker='^', cmap=pl.cm.RdYlGn, s=100)
    pl.scatter(X_train[~idx, 0], X_train[~idx, 1], c=y_train[~idx],
        marker='o', cmap=pl.cm.RdYlGn, s=100)
    pl.arrow(0, 0, 8 * w[0], 8 * w[1], fc='gray', ec='gray',
        head_width=0.5, head_length=0.5)
    pl.text(0, 1, '$w$', fontsize=20)
    pl.arrow(-3, -8, 8 * w[0], 8 * w[1], fc='gray', ec='gray',
        head_width=0.5, head_length=0.5)
    pl.text(-2.6, -7, '$w$', fontsize=20)
    pl.axis('equal')
    pl.show()'''

    # The pairwise transform
    #form all pairwise combinations
    comb = itertools.combinations(range(X_train.shape[0]), 2)
    k = 0
    Xp, yp, diff = [],[],[]
    for (i,j) in comb:
        if y_train[i] == y_train[j] or b_train[i] != b_train[j]:
            continue
        Xp.append(X_train[i]-X_train[j])
        diff.append(y_train[i] - y_train[j])
        yp.append(np.sign(diff[-1]))
        # output balanced classes
        if yp[-1] != (-1) ** k:
            yp[-1] *= -1
            Xp[-1] *= -1
            diff[-1] *= -1
        k += 1
    Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff))
    pl.scatter(Xp[:, 0], Xp[:, 1], c=diff, s=60, marker='o', cmap=pl.cm.Blues)
    x_space = np.linspace(-10, 10)
    pl.plot(x_space * w[1], - x_space * w[0], color='gray')
    pl.text(3, -4, '$\{x^T w = 0\}$', fontsize=17)
    pl.axis('equal')
    pl.show()

    clf = svm.SVC(kernel='linear', C=.1)
    clf.fit(Xp, yp)

    coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)

    '''pl.scatter(X_train[idx, 0], X_train[idx, 1], c=y_train[idx],
        marker='^', cmap=pl.cm.Blues, s=100)
    pl.scatter(X_train[~idx, 0], X_train[~idx, 1], c=y_train[~idx],
        marker='o', cmap=pl.cm.Blues, s=100)
    pl.arrow(0, 0, 7 * coef[0], 7 * coef[1], fc='gray', ec='gray',
        head_width=0.5, head_length=0.5)
    pl.arrow(-3, -8, 7 * coef[0], 7 * coef[1], fc='gray', ec='gray',
        head_width=0.5, head_length=0.5)
    pl.text(1, .7, '$\hat{w}$', fontsize=20)
    pl.text(-2.6, -7, '$\hat{w}$', fontsize=20)
    pl.axis('equal')
    pl.show()'''

    for i in range(100,109):
        tau, _ = stats.kendalltau(
            np.dot(X_test[b_test == i], coef), y_test[b_test == i])
        print('Kendall correlation coefficient for block %s: %.5f' % (i, tau))