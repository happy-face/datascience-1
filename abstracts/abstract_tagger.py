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
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import MLkNN

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
    parser.add_argument("-brsvml", "--binary-relevance-svm-linear", action="store_true", help="Use binary relevance with SVM classifier with linear kernel")
    parser.add_argument("-brsvmr", "--binary-relevance-svm-rbf", action="store_true", help="Use binary relevance with SVM classifier with rbf kernel")
    parser.add_argument("-cclr", "--classifier-chain-logistic-regression", action="store_true", help="Use chain of logistic regression classifiers")
    parser.add_argument("-lplr", "--label-powerset-logistic-regression", action="store_true", help="Use label powerset logistic regression classifier")
    parser.add_argument("-mlknn", "--multi-label-knn", action="store_true", help="Use multi label KNN classifier")
    parser.add_argument("-mr", "--max-rows", type=int, help="Maximum number of samples to use for training (0 - use entire dataset).")
    parser.add_argument("-tmdf", "--title-min-df", type=int, default=3, help="Cutoff document frequency for title words")
    parser.add_argument("-amdf", "--abstract-min-df", type=int, default=5, help="Cutoff document frequency for abstract words")
    parser.add_argument("-nj", "--n-jobs", type=int, default=1, help="n_jobs parameter for GridSearchCV")
    parser.add_argument("-vd", "--validation-data", help="Validation dataset that will be used to select regularization parameters. If not specified regularization parameters will be selected using cross validation")

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
def output_results(file, tag, accuracy_score, classification_report, classifier_details):
    file.write("#\n")
    file.write("# " + tag + "\n")
    file.write("#\n")
    file.write("accuracy score: " + str(accuracy_score) + "\n")
    file.write("\n")
    file.write("classification report:\n")
    file.write(classification_report)
    file.write("\n")

    # write classifier parameters
    file.write("\n")
    file.write("classifier details:\n")
    file.write(classifier_details + "\n")

def output_summary(file, tag, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, best_classifier_details):
    file.write("best_feature_count = %d\n" % best_feature_count)
    file.write("best_feature_ratio = %.1f%%\n" % best_feature_ratio)
    file.write("\n")
    output_results(file, tag, best_accuracy_score, best_classification_report, best_classifier_details)

def get_feature_names(feature_prefix, vectorizer):
    feature_names = vectorizer.get_feature_names()
    return [feature_prefix + s for s in feature_names]


def fit_chi_square_feature_selection(x_train, y_train, feature_names, feature_count):
    from sklearn.feature_selection import SelectKBest, chi2
    ch2 = SelectKBest(chi2, k=feature_count)
    x_train = ch2.fit(x_train, y_train)
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    feature_names = np.asarray(feature_names)
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

    print("Reading train set: %s" % args.train_data, flush=True)
    df_train = pd.read_csv(args.train_data, nrows=args.max_rows)
    df_train.head()

    print("Reading test set: %s" % args.test_data, flush=True)
    df_test = pd.read_csv(args.test_data, nrows=args.max_rows)
    df_test.head()

    df_validation = None
    if args.validation_data:
        print("Reading validation set: %s" % args.validation_data, flush=True)
        df_validation = pd.read_csv(args.validation_data, nrows=args.max_rows)

    print("Normalize training abstracts", flush=True)
    normalize_abstracts(df_train)
    print("Normalize test abstracts", flush=True)
    normalize_abstracts(df_test)
    if df_validation is not None:
        print("Normalize validation abstracts")
        normalize_abstracts(df_validation)

    # main categories is list serialized as string, convert it back
    df_train['main_categories'] = df_train['main_categories'].apply(eval)
    df_test['main_categories'] = df_test['main_categories'].apply(eval)
    if df_validation is not None:
        df_validation['main_categories'] = df_validation['main_categories'].apply(eval)

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
    id_to_category = [0] * len(category_to_id)
    for cat, id in category_to_id.items():
        id_to_category[id] = cat

    print("Generate one hot outputs in training set", flush=True)
    y_df_train = df_train['main_categories'].apply(one_hot_encoder)
    y_df_train = pd.DataFrame(y_df_train.values.tolist(), columns=range(0, len(category_to_id)))

    print("Generate one hot outputs in test set", flush=True)
    y_df_test = df_test['main_categories'].apply(one_hot_encoder)
    y_df_test = pd.DataFrame(y_df_test.values.tolist(), columns=range(0, len(category_to_id)))

    if df_validation is not None:
        print("Generate one hot outputs in validation test set", flush=True)
        y_df_validation = df_validation['main_categories'].apply(one_hot_encoder)
        y_df_validation = pd.DataFrame(y_df_validation.values.tolist(), columns=range(0, len(category_to_id)))

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

    if df_validation is not None:
        x1_validation = df_validation['title'].values
        x2_validation = df_validation['abstract'].values
        y_validation = y_df_validation.values
        x_validation = np.vstack((x1_validation, x2_validation))
        x_validation = x_validation.T

    print("%d documents in train set" %len(x_train), flush=True)
    print("%d documents in test set" % len(x_test), flush=True)
    if df_validation is not None:
        print("%d documents in validation set" % len(x_validation), flush=True)


    print("Extracting features", flush=True)

    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_names = []
    train_feature_groups = []
    test_feature_groups = []
    validation_feature_groups = []

    # title features
    vec_title = None
    if args.title_ngrams > 0:
        vec_title = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,args.title_ngrams), min_df=args.title_min_df)
        train_feature_groups.append(vec_title.fit_transform(x_train[:,0]))
        feature_names += get_feature_names("tit-", vec_title)
        test_feature_groups.append(vec_title.transform(x_test[:,0]))
        if df_validation is not None:
            validation_feature_groups.append(vec_title.transform(x_validation[:,0]))

    # abstract features
    vec_abstract = None
    if args.abstract_ngrams > 0:
        vec_abstract = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,args.abstract_ngrams), min_df=args.abstract_min_df)
        train_feature_groups.append(vec_abstract.fit_transform(x_train[:,1]))
        feature_names += get_feature_names("abs-", vec_abstract)
        test_feature_groups.append(vec_abstract.transform(x_test[:,1]))
        if df_validation is not None:
            validation_feature_groups.append(vec_abstract.transform(x_validation[:,1]))

    assert len(train_feature_groups) > 0
    assert len(test_feature_groups) > 0
    if df_validation is not None:
        assert len(validation_feature_groups) > 0

    x_train = scipy.sparse.hstack(tuple(train_feature_groups))
    x_test = scipy.sparse.hstack(tuple(test_feature_groups))
    x_validation = None
    if df_validation is not None:
        x_validation = scipy.sparse.hstack(tuple(validation_feature_groups))

    print("Total feature count: %d" % len(feature_names), flush=True)

    best_accuracy_score = 0.0
    best_feature_ratio = 0.0
    best_feature_count = 0
    best_classification_report = ""
    best_classifier_detals = ""
    best_train_accuracy_score =0.0
    best_train_classification_report = ""
    best_validation_accuracy_score = 0.0
    best_validation_classification_report = ""

    for feature_ratio in args.feature_selection_ratio:
        # selecting the best k features from the data set
        feature_count = int(feature_ratio * len(feature_names))
        print("Extracting %.1f%% best features by a chi-squared test" % (feature_ratio * 100))
        ch2, feature_names_sel = fit_chi_square_feature_selection(x_train, y_train, feature_names, feature_count)
        x_train_sel = ch2.transform(x_train)
        x_test_sel = ch2.transform(x_test)
        if x_validation is not None:
            x_validation_sel = ch2.transform(x_validation)

        gridsearch_cv = None
        gridsearch_refit = True
        if x_validation is not None:
            x_gridsearch = scipy.sparse.vstack((x_train_sel, x_validation_sel))
            y_gridsearch = scipy.sparse.vstack([scipy.sparse.csr_matrix(y_train), scipy.sparse.csr_matrix(y_validation)])
            x_train_ids = range(0, np.shape(x_train_sel)[0])
            x_validation_ids = range(np.shape(x_train_sel)[0], np.shape(x_train_sel)[0] + np.shape(x_validation_sel)[0])
            gridsearch_cv = [(list(x_train_ids), list(x_validation_ids))]

            # we shouldnt refit after GridSEarchCV because we shouldn't include validation data into training
            gridsearch_refit = False

        else:
            x_gridsearch = x_train_sel
            y_gridsearch = y_train

        classifier_details = "NA"
        if args.binary_relevance_naive_bayes:
            print("Train binary relevance naive bayes tagger")
            classifier = BinaryRelevance(GaussianNB())
            classifier.fit(x_train_sel, y_train)

        elif args.binary_relevance_logistic_regression:
            print("Train binary relevance logistic regression tagger")
            if len(args.classifier_C) > 1:
                parameters = [
                    {
                        'classifier': [LogisticRegression(solver='liblinear')],
                        'classifier__C': args.classifier_C,
                    }
                ]
                # iid = True : use average across folds as selection criteria
                # refit = True : fit model on all data after getting best parameters with CV
                gridSearch = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy', iid=True, refit=gridsearch_refit, n_jobs=args.n_jobs, cv = gridsearch_cv)
                gridSearch.fit(x_gridsearch, y_gridsearch)
                classifier_details = "best_params = " + str(gridSearch.best_params_)
                classifier = gridSearch.best_estimator_
            else:
                classifier = BinaryRelevance(LogisticRegression(solver='liblinear', C = args.classifier_C[0]))
                classifier.fit(x_train_sel, y_train)

        elif args.binary_relevance_svm_linear or args.binary_relevance_svm_rbf:
            print("Train binary relevance SVM regression tagger")
            svm_kernel = 'linear'
            if args.binary_relevance_svm_rbf:
                svm_kernel = 'rbf'

            if len(args.classifier_C) > 1:
                parameters = [
                    {
                        'classifier': [SVC(kernel=svm_kernel)],
                        'classifier__C': args.classifier_C,
                    }
                ]
                # iid = True : use average across folds as selection criteria
                # refit = True : fit model on all data after getting best parameters with CV
                gridSearch = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy', iid=True, refit=gridsearch_refit, n_jobs=args.n_jobs, cv = gridsearch_cv)
                gridSearch.fit(x_gridsearch, y_gridsearch)
                classifier_details = "best_params = " + str(gridSearch.best_params_)
                classifier = gridSearch.best_estimator_
            else:
                classifier = BinaryRelevance(SVC(kernel=svm_kernel, C = args.classifier_C[0]))
                classifier.fit(x_train_sel, y_train)

        elif args.classifier_chain_logistic_regression:
            print("Train classifier chain logistic regression tagger")
            if len(args.classifier_C) > 1:
                parameters = [
                    {
                        'classifier': [LogisticRegression(solver='liblinear')],
                        'classifier__C': args.classifier_C,
                    }
                ]
                # iid = True : use average across folds as selection criteria
                # refit = True : fit model on all data after getting best parameters with CV
                gridSearch = GridSearchCV(ClassifierChain(), parameters, scoring='accuracy', iid=True, refit=gridsearch_refit, n_jobs=args.n_jobs, cv = gridsearch_cv)
                gridSearch.fit(x_gridsearch, y_gridsearch)
                classifier_details = "best_params = " + str(gridSearch.best_params_)
                classifier = gridSearch.best_estimator_
            else:
                classifier = ClassifierChain(LogisticRegression(solver='liblinear', C = args.classifier_C[0]))
                classifier.fit(x_train_sel, y_train)

        elif args.label_powerset_logistic_regression:
            print("Train label powerset logistic regression tagger")
            if len(args.classifier_C) > 1:
                parameters = [
                    {
                        'classifier': [LogisticRegression(solver='liblinear')],
                        'classifier__C': args.classifier_C,
                    }
                ]
                # iid = True : use average across folds as selection criteria
                # refit = True : fit model on all data after getting best parameters with CV
                gridSearch = GridSearchCV(LabelPowerset(), parameters, scoring='accuracy', iid=True, refit=gridsearch_refit, n_jobs=args.n_jobs, cv = gridsearch_cv)
                gridSearch.fit(x_gridsearch, y_gridsearch)
                classifier_details = "best_params = " + str(gridSearch.best_params_)
                classifier = gridSearch.best_estimator_
            else:
                classifier = LabelPowerset(LogisticRegression(solver='liblinear', C = args.classifier_C[0]))
                classifier.fit(x_train_sel, y_train)


        elif args.multi_label_knn:
            print("Train multi label KNN classifier")
            parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
            gridSearch = GridSearchCV(MLkNN(), parameters, scoring='accuracy', iid=True, refit=gridsearch_refit, n_jobs=args.n_jobs, cv = gridsearch_cv)
            gridSearch.fit(x_gridsearch, y_gridsearch)
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
        classification_report = metrics.classification_report(y_test, predictions, target_names=id_to_category)

        # evaluate training set with the model
        predictions = classifier.predict(x_train_sel.astype(float))
        predictions = predictions.todense()
        # score evaluation results
        train_accuracy_score = metrics.accuracy_score(y_train, predictions)
        train_classification_report = metrics.classification_report(y_train, predictions, target_names=id_to_category)

        if x_validation is not None:
            # evaluate training set with the model
            predictions = classifier.predict(x_validation_sel.astype(float))
            predictions = predictions.todense()
            # score evaluation results
            validation_accuracy_score = metrics.accuracy_score(y_validation, predictions)
            validation_classification_report = metrics.classification_report(y_validation, predictions, target_names=id_to_category)

        # output scores to stdout
        output_results(sys.stdout, "TEST SET", accuracy_score, classification_report, classifier_details)
        output_results(sys.stdout, "TRAINING SET", train_accuracy_score, train_classification_report, classifier_details)
        if x_validation is not None:
            output_results(sys.stdout, "VALIDATION SET", validation_accuracy_score, validation_classification_report, classifier_details)
        print()

        # output scores to file
        with open(os.path.join(args.output, "results_%d.txt" % int(100 * feature_ratio)), 'w') as results_file:
            output_results(results_file, "TEST SET", accuracy_score, classification_report, classifier_details)
            output_results(results_file, "TRAINING SET", train_accuracy_score, train_classification_report, classifier_details)
            if x_validation is not None:
                output_results(results_file, "VALIDATION SET", validation_accuracy_score, validation_classification_report, classifier_details)

        # update best results if needed
        if (accuracy_score > best_accuracy_score):
            best_accuracy_score = accuracy_score
            best_feature_ratio = feature_ratio
            best_feature_count = feature_count
            best_classification_report = classification_report
            best_classifier_details = classifier_details
            best_train_accuracy_score = train_accuracy_score
            best_train_classification_report = train_classification_report
            if x_validation is not None:
                best_validation_accuracy_score = validation_accuracy_score
                best_validation_classification_report = validation_classification_report

        # store current model
        model_path = os.path.join(args.output, "model_%d.pickle" % int(100 * feature_ratio))
        store_model(model_path, vec_abstract, vec_title, ch2, classifier, category_to_id)

        sys.stdout.flush()

    print()
    print()
    print("=== summary ===")
    output_summary(sys.stdout, "TEST SET", best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, best_classifier_details)
    output_summary(sys.stdout, "TRAINING SET", best_feature_count, best_feature_ratio, best_train_accuracy_score, best_train_classification_report, best_classifier_details)
    if x_validation is not None:
        output_summary(sys.stdout, "VALIDATION SET", best_feature_count, best_feature_ratio, best_validation_accuracy_score, best_validation_classification_report, best_classifier_details)

    with open(os.path.join(args.output, "summary.txt"), 'w') as summary_file:
        output_summary(summary_file, "TEST SET", best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, best_classifier_details)
        output_summary(summary_file, "TRAINING SET", best_feature_count, best_feature_ratio, best_train_accuracy_score, best_train_classification_report, best_classifier_details)
        if x_validation is not None:
            output_summary(summary_file, "VALIDATION SET", best_feature_count, best_feature_ratio, best_validation_accuracy_score, best_validation_classification_report, best_classifier_details)
