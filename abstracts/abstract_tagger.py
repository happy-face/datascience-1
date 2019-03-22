# -*- coding: utf-8 -*-

import argparse
import os
import operator
import scipy
import numpy as np
import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train-data", required=True, help="Input train data CSV")
    parser.add_argument("-te", "--test-data", required=True, help="Input test data CSV")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    parser.add_argument("--binary-relevance-naive-bayes", action="store_true", help="Use binary relevance with naive bayes classifier")
    parser.add_argument("--binary-relevance-logistic-regression", action="store_true", help="Use binary relevance with logistic regression classifier")
    parser.add_argument("-ms", "--max-samples", type=int, help="Maximum number of samples to use for training (0 - use entire dataset).")

    return parser.parse_args()

def output_results(file, accuracy_score, classification_report, category_to_id, classifier_details):
    file.write("accuracy score: " + str(accuracy_score) + "\n")
    file.write("\n")
    file.write("classification report:\n")
    file.write(classification_report)
    file.write("\n")

    # we want to sort by value - ID
    category_id_pairs = sorted(category_to_id.items(), key=operator.itemgetter(1))
    for category, id in category_id_pairs:
        file.write("%i\t%s\n" % (id, category))

    # write classifier parameters
    file.write("\n")
    file.write("classifier details:\n")
    file.write(classifier_details)

def output_summary(file, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details):
    file.write("best_feature_count = %d\n" % best_feature_count)
    file.write("best_feature_ratio = %.1f%%\n" % best_feature_ratio)
    file.write("\n")
    output_results(file, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details)

def get_feature_names(feature_prefix, vectorizer):
    feature_names = vectorizer.get_feature_names()
    return [feature_prefix + s for s in feature_names]


def chi_square_feature_selection(x_train, y_train, x_test, feature_names, feature_count):
    from sklearn.feature_selection import SelectKBest, chi2
    ch2 = SelectKBest(chi2, k=feature_count)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test = ch2.transform(x_test)
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    return x_train, x_test, feature_names


def normalize_abstracts(df):
    stopWordList = stopwords.words('english')
    lemma = WordNetLemmatizer()
    token = ToktokTokenizer()

    def remove_special_chars(text):
        str_chars = '`-=~@#$%^&*()_+[!{;”:\’><.,/?”}]'
        for w in text:
            if w in str_chars:
                text = text.replace(w,'')
        return text

    def lemitize_words(text):
        words = token.tokenize(text)
        listLemma =[]
        for w in words:
            x = lemma.lemmatize(w,'v')
            listLemma.append(x)
        text = " ".join(listLemma)
        return text

    def stop_words_remove(text):
        wordList = [x.strip() for x in token.tokenize(text)]
        removedList = [x for x in wordList if not x in stopWordList]
        text =" ".join(removedList)
        return text

    def preprocessing_text(text):
        text = remove_special_chars(text)
        text = lemitize_words(text)
        text = stop_words_remove(text)
        return text

    df['abstract'] = df['abstract'].map(lambda x: preprocessing_text(x))


#
# training
#   - preprocessing
#   - extracting features (fit)
#   - selecting features (fit)
#   - model (fit)
#
#
#
#

if __name__ == "__main__":
    args = parse_args()

    # create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("Reading train set: %s" % args.train_data)
    df_train = pd.read_csv(args.train_data, nrows=args.max_samples)
    df_train.head()

    print("Reading test set: %s" % args.test_data)
    df_test = pd.read_csv(args.test_data, nrows=args.max_samples)
    df_test.head()

    print("Normalizing abstracts")
    normalize_abstracts(df_train)
    normalize_abstracts(df_test)

    # main categories is list serialized as string, convert it back
    df_train['main_categories'] = df_train['main_categories'].apply(eval)
    df_test['main_categories'] = df_test['main_categories'].apply(eval)

    #apply one-hot encoding for Binary Relevance
    def one_hot_encoder(tags):
        vec = [0] * len(category_to_id)
        for tag in tags:
            vec[category_to_id[tag]]=1
        return vec

    #Map main categories to integers
    unique_categories = set()
    for n in df_train.main_categories:
        print n
        unique_categories.update(n)
    print unique_categories
    category_to_id = dict([(j,i) for i, j in enumerate(sorted(unique_categories))])


    y_df_train = df_train['main_categories'].apply(one_hot_encoder)
    y_df_train = pd.DataFrame(y_df_train.values.tolist(), columns=range(0, len(category_to_id)))

    y_df_test = df_test['main_categories'].apply(one_hot_encoder)
    y_df_test = pd.DataFrame(y_df_test.values.tolist(), columns=range(0, len(category_to_id)))

    x1_train = df_train['title'].values
    x2_train = df_train['abstract'].values
    y_train = y_df_train.values
    x_train = np.vstack((x1_train, x2_train))
    x_train = x_train.T

    x1_test = df_test['title'].values
    x2_test = df_test['abstract'].values
    y_test = y_df_test.values
    x_test = np.vstack((x1_test, x2_test))
    x_test = x_test.T

    print("%d documents in train set" %len(x_train))
    print("%d documents in test set" % len(x_test))

    print("Extracting features")

    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_names = []

    # title features
    vec_title = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,2))
    title_train = vec_title.fit_transform(x_train[:,0])
    feature_names += get_feature_names("tit-", vec_title)
    title_test = vec_title.transform(x_test[:,0])

    # abstract features
    vec_abstract = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,2))
    abstract_train = vec_abstract.fit_transform(x_train[:,1])
    feature_names += get_feature_names("abs-", vec_abstract)
    abstract_test = vec_abstract.transform(x_test[:,1])

    x_train = scipy.sparse.hstack((title_train, abstract_train))
    x_test = scipy.sparse.hstack((title_test, abstract_test))

    print("Total feature count: %d" % len(feature_names))

    best_accuracy_score = 0.0
    best_feature_ratio = 0.0
    best_feature_count = 0
    best_classification_report = ""
    best_classifier_detals = ""

    for i in range(0, 10):
        # selecting the best k features from the data set
        feature_ratio = float(i + 1) / 10.0
        feature_count = int(((i + 1) * len(feature_names)) / 10)
        print("Extracting %.1f%% best features by a chi-squared test" % feature_ratio)
        x_train_sel, x_test_sel, feature_names_sel = chi_square_feature_selection(x_train, y_train, x_test, feature_names, feature_count)

        print("Training Binary Relevance classifier")

        classifier_details = "NA"
        if args.binary_relevance_naive_bayes:
            classifier = BinaryRelevance(GaussianNB())
            classifier.fit(x_train_sel, y_train)

        elif args.binary_relevance_logistic_regression:
            parameters = [
                {
                    'classifier': [LogisticRegression(solver='lbfgs')],
                    'classifier__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                }
            ]
            classifier = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy', iid=True)
            classifier.fit(x_train_sel, y_train)
            classifier_details = "best_params = " + str(classifier.best_params_)
        else:
            print("ERROR: specify classification model")
            exit()


        predictions = classifier.predict(x_test_sel.astype(float))
        predictions = predictions.todense()

        # conf_mat = metrics.confusion_matrix(x_test[:,1], predictions[:,1])

        accuracy_score = metrics.accuracy_score(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)

        # output results to file for this feature count experiment
        output_results(sys.stdout, accuracy_score, classification_report, category_to_id, classifier_details)
        with open(os.path.join(args.output, "results_%d.txt" % int(100 * feature_ratio)), 'w') as results_file:
            output_results(results_file, accuracy_score, classification_report, category_to_id, classifier_details)
        print()

        # update best results if needed
        if (accuracy_score > best_accuracy_score):
            best_accuracy_score = accuracy_score
            best_feature_ratio = feature_ratio
            best_feature_count = feature_count
            best_classification_report = classification_report
            best_classifier_details = classifier_details

    print()
    print()
    print("=== summary ===")
    output_summary(sys.stdout, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details)
    with open(os.path.join(args.output, "summary.txt"), 'w') as summary_file:
        output_summary(summary_file, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details)
