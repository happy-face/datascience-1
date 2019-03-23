# -*- coding: utf-8 -*-

import argparse
import os
import operator
import scipy
import numpy as np
import sys
import pickle

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
    parser.add_argument("-ang", "--abstract-ngrams", type=int, default=2, help="Abstract ngram features that should be used. Specify 0 to turn off abstract features.")
    parser.add_argument("-tng", "--title-ngrams", type=int, default=2, help="Title ngram features that should be used. Specify 0 to turn off title features.")
    parser.add_argument("-fsr", "--feature-selection-ratio", nargs='+', type=float, default=[1.0], help="Ratio of features which should be keept in feature selection.")
    parser.add_argument("-cc", "--classifier-C", nargs='+', type=float, default = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], help="Regularization factors that will be passed to GridSearchCV sklearn.")
    parser.add_argument("-brnb", "--binary-relevance-naive-bayes", action="store_true", help="Use binary relevance with naive bayes classifier")
    parser.add_argument("-brlr", "--binary-relevance-logistic-regression", action="store_true", help="Use binary relevance with logistic regression classifier")
    parser.add_argument("-mr", "--max-rows", type=int, help="Maximum number of samples to use for training (0 - use entire dataset).")
    parser.add_argument("-tmdf", "--title-min-df", type=int, default=1, help="Cutoff document frequency for title words")
    parser.add_argument("-amdf", "--abstract-min-df", type=int, default=2, help="Cutoff document frequency for abstract words")

    return parser.parse_args()


#
# Serialization / deserialization of the model
#
def store_model(path, abstract_featurizer, title_featurizer, feature_selection, classifier, category_to_id):
    featurizer = {"abstract_featurizer" : abstract_featurizer, "title_featurizer": title_featurizer}

    model = {"featurizer": featurizer}
    model["featurizer"] = featurizer
    model["feature_selection"] = feature_selection
    model["classifier"] = classifier
    model["category_to_id"] = category_to_id
    model["id_to_category"] = {v: k for k, v in category_to_id.items()}
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(path)
        abstract_featurizer = model["featurizer"]["abstract_featurizer"]
        title_featurizer = model["featurizer"]["title_featurizer"]
        feature_selection = model["feature_selection"]
        classifier = model["classifier"]
        category_to_id = model["category_to_id"]
        id_to_category = model["id_to_category"]
        
        return abstract_featurizer, title_featurizer, feature_selection, classifier, category_to_id, id_to_category


#
# Pringing statistics and results
#
def output_results(file, tag, accuracy_score, classification_report, category_to_id, classifier_details):
    file.write("#\n")
    file.write("# " + tag + "\n")
    file.write("#\n")
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
    file.write(classifier_details + "\n")

def output_summary(file, tag, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details):
    file.write("best_feature_count = %d\n" % best_feature_count)
    file.write("best_feature_ratio = %.1f%%\n" % best_feature_ratio)
    file.write("\n")
    output_results(file, tag, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details)

def get_feature_names(feature_prefix, vectorizer):
    feature_names = vectorizer.get_feature_names()
    return [feature_prefix + s for s in feature_names]


def fit_chi_square_feature_selection(x_train, y_train, feature_names, feature_count):
    from sklearn.feature_selection import SelectKBest, chi2
    ch2 = SelectKBest(chi2, k=feature_count)
    x_train = ch2.fit(x_train, y_train)
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    feature_names = np.asarray(feature_names)
    print(feature_names)
    return ch2, feature_names


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


if __name__ == "__main__":
    args = parse_args()

    # create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("Reading train set: %s" % args.train_data)
    df_train = pd.read_csv(args.train_data, nrows=args.max_rows)
    df_train.head()

    print("Reading test set: %s" % args.test_data)
    df_test = pd.read_csv(args.test_data, nrows=args.max_rows)
    df_test.head()

    print("Normalize training abstracts")
    normalize_abstracts(df_train)
    print("Normalize test abstracts")
    normalize_abstracts(df_test)

    # main categories is list serialized as string, convert it back
    df_train['main_categories'] = df_train['main_categories'].apply(eval)
    df_test['main_categories'] = df_test['main_categories'].apply(eval)

    #apply one-hot encoding for Binary Relevance
    def one_hot_encoder(tags):
        vec = [0] * len(category_to_id)
        for tag in tags:
            if tag in category_to_id:
                vec[category_to_id[tag]]=1
            else:
                print("WARNING: ignoring tag %s", tag)
        return vec

    #Map main categories to integers
    unique_categories = set()
    for n in df_train.main_categories:
        unique_categories.update(n)
    category_to_id = dict([(j,i) for i, j in enumerate(sorted(unique_categories))])

    print("Generate one hot outputs in training set")
    y_df_train = df_train['main_categories'].apply(one_hot_encoder)
    y_df_train = pd.DataFrame(y_df_train.values.tolist(), columns=range(0, len(category_to_id)))

    print("Generate one hot outputs in test set")
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
    train_feature_groups = []
    test_feature_groups = []

    # title features
    vec_title = None
    if args.title_ngrams > 0:
        vec_title = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,args.title_ngrams), min_df=args.title_min_df)
        train_feature_groups.append(vec_title.fit_transform(x_train[:,0]))
        feature_names += get_feature_names("tit-", vec_title)
        test_feature_groups.append(vec_title.transform(x_test[:,0]))

    # abstract features
    vec_abstract = None
    if args.abstract_ngrams > 0:
        vec_abstract = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,args.abstract_ngrams), min_df=args.abstract_min_df)
        train_feature_groups.append(vec_abstract.fit_transform(x_train[:,1]))
        feature_names += get_feature_names("abs-", vec_abstract)
        test_feature_groups.append(vec_abstract.transform(x_test[:,1]))

    assert len(train_feature_groups) > 0
    assert len(test_feature_groups) > 0

    x_train = scipy.sparse.hstack(tuple(train_feature_groups))
    x_test = scipy.sparse.hstack(tuple(test_feature_groups))

    print("Total feature count: %d" % len(feature_names))
    sys.stdout.flush()

    best_accuracy_score = 0.0
    best_feature_ratio = 0.0
    best_feature_count = 0
    best_classification_report = ""
    best_classifier_detals = ""
    best_train_accuracy_score =0.0
    best_train_classification_report = ""

    for feature_ratio in args.feature_selection_ratio:
        # selecting the best k features from the data set
        feature_count = int(feature_ratio * len(feature_names))
        print("Extracting %.1f%% best features by a chi-squared test" % (feature_ratio * 100))
        ch2, feature_names_sel = fit_chi_square_feature_selection(x_train, y_train, feature_names, feature_count)
        x_train_sel = ch2.transform(x_train)
        x_test_sel = ch2.transform(x_test)
        

        classifier_details = "NA"
        if args.binary_relevance_naive_bayes:
            print("Train binary relevance naive bayes tagger")
            classifier = BinaryRelevance(GaussianNB())
            classifier.fit(x_train_sel, y_train)

        elif args.binary_relevance_logistic_regression:
            print("Train binary relevance logistic regression tagger")
            parameters = [
                {
                    'classifier': [LogisticRegression()],
                    'classifier__C': args.classifier_C,
                }
            ]
            # iid = True : use average across folds as selection criteria
            # refit = True : fit model on all data after getting best parameters with CV
            gridSearch = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy', iid=True, refit=True, n_jobs=-1)
            gridSearch.fit(x_train_sel, y_train)
            classifier_details = "best_params = " + str(gridSearch.best_params_)
            classifier = gridSearch.best_estimator_

        else:
            print("ERROR: specify classification model")
            exit()


        # evaluate test set with the model
        predictions = classifier.predict(x_test_sel.astype(float))
        predictions = predictions.todense()
        # score evaluation results
        accuracy_score = metrics.accuracy_score(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
        
        # evaluate training set with the model
        predictions = classifier.predict(x_train_sel.astype(float))
        predictions = predictions.todense()
        # score evaluation results
        train_accuracy_score = metrics.accuracy_score(y_train, predictions)
        train_classification_report = metrics.classification_report(y_train, predictions)       

        # output scores to stdout
        output_results(sys.stdout, "TEST SET", accuracy_score, classification_report, category_to_id, classifier_details)
        output_results(sys.stdout, "TRAINING SET", train_accuracy_score, train_classification_report, category_to_id, classifier_details)
        print()

        # output scores to file
        with open(os.path.join(args.output, "results_%d.txt" % int(100 * feature_ratio)), 'w') as results_file:
            output_results(results_file, "TEST SET", train_accuracy_score, classification_report, category_to_id, classifier_details)
            output_results(results_file, "TRAINING SET", accuracy_score, train_classification_report, category_to_id, classifier_details)
        
        # update best results if needed
        if (accuracy_score > best_accuracy_score):
            best_accuracy_score = accuracy_score
            best_feature_ratio = feature_ratio
            best_feature_count = feature_count
            best_classification_report = classification_report
            best_classifier_details = classifier_details
            best_train_accuracy_score = train_accuracy_score
            best_train_classification_report = train_classification_report
            
        # store current model
        model_path = os.path.join(args.output, "model_%d.pickle" % int(100 * feature_ratio))
        store_model(model_path, vec_abstract, vec_title, ch2, classifier, category_to_id)
        
        sys.stdout.flush()

    print()
    print()
    print("=== summary ===")
    output_summary(sys.stdout, "TEST SET", best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details)
    output_summary(sys.stdout, "TRAINING SET", best_feature_count, best_feature_ratio, best_train_accuracy_score, best_train_classification_report, category_to_id, best_classifier_details)

    with open(os.path.join(args.output, "summary.txt"), 'w') as summary_file:
        output_summary(summary_file, "TEST SET", best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id, best_classifier_details)
        output_summary(summary_file, "TRAINING SET", best_feature_count, best_feature_ratio, best_train_accuracy_score, best_train_classification_report, category_to_id, best_classifier_details)