import os
import itertools
import numpy as np
import argparse
from scipy import stats
import pylab as pl
import cv2
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd



# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV dataset file")
    parser.add_argument("-l", "--labels", required=True, help = "Input CSV labels file")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    return parser.parse_args()


#
# Converts one row of feature.csv table into feature vector
#
def row_to_feature_vector(dataFrame, row_id):
    fv = []
    fv.append(dataFrame.blur[row_id])
    fv.append(dataFrame.noise[row_id])
    fv.append(dataFrame.brightness[row_id])
    return fv


#
# Reads labels CSV into dictonary mapping im_path to (label, set) pair
#
def read_labels(path, label2id):
    img2label_and_set = {}
    lab_df = pd.read_csv(args.labels, ',')
    for row_id in range(0, len(lab_df.im_path)):
        im_path = lab_df.im_path[row_id]
        assert im_path not in img2label_and_set
        img2label_and_set[im_path] = (label2id[lab_df.label[row_id]], lab_df.set_name[row_id])
    return img2label_and_set


#
# Class representing information about image: feature vector, label and im_path
#
class ImageSample:
    clf = None

    def __init__(self, X, y, im_path):
        self.X = X
        self.y = y
        self.im_path = im_path

    def __str__(self):
        return "\t".join([self.im_path, str(self.X), str(self.y)])

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __lt__(self, other):
        return ImageSample.clf.predict([self.X + other.X])[0] == -1

    def __le__(self, other):
        return self.__lt__(other)

    def __gt__(self, other):
        return ImageSample.clf.predict([self.X + other.X])[0] == 1

    def __ge__(self, other):
        return self.__gt__(other)

#    def __repr__(self):
#        return str(self)


#
# Creates collections of ImageSamples for each unique set
#
def create_sets(feat_df, img2label_and_set):
    set_name2image_samples = {}
    for row_id in range(0, len(feat_df.im_path)):
        im_path = feat_df.im_path[row_id]
        if not im_path in img2label_and_set:
            print("Skipping %s because label was not found" % im_path)
            continue
        label, set_name = img2label_and_set[im_path]

        X = row_to_feature_vector(feat_df, row_id)

        if not set_name in set_name2image_samples:
            set_name2image_samples[set_name] = []
        set_name2image_samples[set_name].append(ImageSample(X, label, im_path))

    return set_name2image_samples


#
# For debugging purposes
#
def print_sets(set_name2image_samples):
    for set_name in set_name2image_samples.keys():
        print(set_name)
        for img_sample in set_name2image_samples[set_name]:
            print("\t", str(img_sample))
        print()


#
# Converts list of feature vectors and labels into
# list of feature vector pairs (concatenatin) and diff labels.
#
# X, y - input feature vectors and labels
# Xp, yp - output pairwise transformed feature vectors and labels
#
def pairwise_transform(X, y, Xp, yp):
    assert len(X) == len(y)
    assert len(Xp) == len(yp)

    comb = itertools.combinations(range(len(X)), 2)
    for (i,j) in comb:
        # skip equal labels because we don't have equal diff class
        if y[i] == y[j]:
            continue

        # note: + is list concatenation
        Xp.append(X[i] + X[j])
        yp.append(np.sign(y[i] - y[j]))

        # reverse order for balanced classes
        Xp.append(X[j] + X[i])
        yp.append(np.sign(y[j] - y[i]))


def sets2pairwise(set_image_samples):
    Xp = []
    yp = []
    for set_name, image_samples in set_image_samples:
        set_X = []
        set_y = []
        for image_sample in image_samples:
            set_X.append(image_sample.X)
            set_y.append(image_sample.y)
        pairwise_transform(set_X, set_y, Xp, yp)
    return Xp, yp


def basic_sort(in_list):
    r = list(in_list)
    for i in range(0, len(r)):
        for j in range(i + 1, len(r)):
            if r[i] < r[j]:
                t = r[j]
                r[j] = r[i]
                r[i] = t
    return r


def score_top_1(set2sorted):
    correct = 0
    for set_name, image_samples in set2sorted.items():
        if image_samples[0].y == 1:
            correct += 1
    return float(correct) / len(set2sorted)


label2id = {"discard": 0, "keep": 1}


if __name__ == "__main__":
    np.random.seed(31415)

    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load labels
    img2label_and_set = read_labels(args.labels, label2id)

    # featurize
    feat_df = pd.read_csv(args.input)
    set_name2image_samples = create_sets(feat_df, img2label_and_set)
    print_sets(set_name2image_samples)

    # training/test split
    set_image_sample_items = list(set_name2image_samples.items())
    train, test = train_test_split(set_image_sample_items, test_size=0.5)

    # transform training into pairwise classification problem
    Xp_train, yp_train = sets2pairwise(train)
    Xp_test, yp_test = sets2pairwise(test)

    clf = svm.SVC(kernel='linear', C=.1, verbose=True)
    clf.fit(Xp_train, yp_train)
    print()
    print()
    print("train accuracy: %.2f%%" % (100.0 * clf.score(Xp_train, yp_train)))
    print("test accuracy: %.2f%%" % (100.0 * clf.score(Xp_test, yp_test)))


    # set reference to classifier so that we can use comparison methods on ImageSample
    ImageSample.clf = clf

    # rank all sets in training
    set2sorted_train = {}
    for set_name, image_samples in train:
        set2sorted_train[set_name] = sorted(image_samples, reverse=True)

    #rank all sets in test
    set2sorted_test = {}
    for set_name, image_samples in test:
        set2sorted_test[set_name] = sorted(image_samples, reverse=True)

    print()
    print()
    print("train top1: %.2f%%" % (100 * score_top_1(set2sorted_train)))
    print("test top1: %.2f%%" % (100 * score_top_1(set2sorted_test)))


    # X = df[['im_file', 'blur', 'noise', 'brightness']]
    # X['brightness'] = X['brightness'].str.strip('[]').astype(float)
    # X = X.sort_values('im_file')
    # X = X.drop(columns=['im_file'])
    # X = X.reset_index(drop=True)
    #
    # y_df = pd.read_csv(args.labels, '\t')
    # y_df = y_df[0:(X.shape[0])]
    # y = y_df['Unnamed: 1']
    #
    # blocks = df['set']
    #
    # sss = StratifiedShuffleSplit(n_splits=2, test_size=.5)
    #
    # for train_index, test_index in sss.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     b_train, b_test = blocks.iloc[train_index], blocks.iloc[test_index]
    #
    #
    # X_train = X_train.values
    # X_test = X_test.values
    # y_train = y_train.values
    # y_test = y_test.values
    # b_train = b_train.values
    # b_test = b_test.values
    #
    # # plot the result
    # '''idx = (b_train == 0)
    # pl.scatter(X_train[idx, 0], X_train[idx, 1], c=y_train[idx],
    #     marker='^', cmap=pl.cm.RdYlGn, s=100)
    # pl.scatter(X_train[~idx, 0], X_train[~idx, 1], c=y_train[~idx],
    #     marker='o', cmap=pl.cm.RdYlGn, s=100)
    # pl.arrow(0, 0, 8 * w[0], 8 * w[1], fc='gray', ec='gray',
    #     head_width=0.5, head_length=0.5)
    # pl.text(0, 1, '$w$', fontsize=20)
    # pl.arrow(-3, -8, 8 * w[0], 8 * w[1], fc='gray', ec='gray',
    #     head_width=0.5, head_length=0.5)
    # pl.text(-2.6, -7, '$w$', fontsize=20)
    # pl.axis('equal')
    # pl.show()'''
    #
    # # The pairwise transform
    # #form all pairwise combinations
    # comb = itertools.combinations(range(X_train.shape[0]), 2)
    # k = 0
    # Xp, yp, diff = [],[],[]
    # for (i,j) in comb:
    #     if y_train[i] == y_train[j] or b_train[i] != b_train[j]:
    #         continue
    #     Xp.append(X_train[i]-X_train[j])
    #     diff.append(y_train[i] - y_train[j])
    #     yp.append(np.sign(diff[-1]))
    #     # output balanced classes
    #     if yp[-1] != (-1) ** k:
    #         yp[-1] *= -1
    #         Xp[-1] *= -1
    #         diff[-1] *= -1
    #     k += 1
    # Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff))
    # pl.scatter(Xp[:, 0], Xp[:, 1], c=diff, s=60, marker='o', cmap=pl.cm.Blues)
    # x_space = np.linspace(-10, 10)
    # pl.plot(x_space * w[1], - x_space * w[0], color='gray')
    # pl.text(3, -4, '$\{x^T w = 0\}$', fontsize=17)
    # pl.axis('equal')
    # pl.show()
    #
    # clf = svm.SVC(kernel='linear', C=.1)
    # clf.fit(Xp, yp)
    #
    # coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)
    #
    # '''pl.scatter(X_train[idx, 0], X_train[idx, 1], c=y_train[idx],
    #     marker='^', cmap=pl.cm.Blues, s=100)
    # pl.scatter(X_train[~idx, 0], X_train[~idx, 1], c=y_train[~idx],
    #     marker='o', cmap=pl.cm.Blues, s=100)
    # pl.arrow(0, 0, 7 * coef[0], 7 * coef[1], fc='gray', ec='gray',
    #     head_width=0.5, head_length=0.5)
    # pl.arrow(-3, -8, 7 * coef[0], 7 * coef[1], fc='gray', ec='gray',
    #     head_width=0.5, head_length=0.5)
    # pl.text(1, .7, '$\hat{w}$', fontsize=20)
    # pl.text(-2.6, -7, '$\hat{w}$', fontsize=20)
    # pl.axis('equal')
    # pl.show()'''
    #
    # for i in range(100,109):
    #     tau, _ = stats.kendalltau(
    #         np.dot(X_test[b_test == i], coef), y_test[b_test == i])
    #     print('Kendall correlation coefficient for block %s: %.5f' % (i, tau))
