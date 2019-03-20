# -*- coding: utf-8 -*-

import argparse
import os
import operator
import scipy
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV dataset file")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    parser.add_argument("--binary-relevance-naive-bayes", action="store_true", help="Use binary relevance with naive bayes classifier")
    parser.add_argument("--binary-relevance-logistic-regression", action="store_true", help="Use binary relevance with logistic regression classifier")
    parser.add_argument("-ms", "--max-samples", type=int, help="Maximum number of samples to use for training (0 - use entire dataset).")

    return parser.parse_args()


def output_results(file, accuracy_score, classification_report, category_to_id):
    file.write("accuracy score: " + str(accuracy_score) + "\n")
    file.write("\n")
    file.write("classification report:\n")
    file.write(classification_report)
    file.write("\n")

    # we want to sort by value - ID
    category_id_pairs = sorted(category_to_id.items(), key=operator.itemgetter(1))
    for category, id in category_id_pairs:
        file.write("%i\t%s\n" % (id, category))


def output_summary(file, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id):
    file.write("best_feature_count = %d\n" % best_feature_count)
    file.write("best_feature_ratio = %.1f%%\n" % best_feature_ratio)
    file.write("\n")
    output_results(file, best_accuracy_score, best_classification_report, category_to_id)


#
# Adds main_categories column
#
def generate_main_categories(df):
    # Extract main categories from tag strings
    def main_categories(tags):
        main_tags = [i.split(".") for i in tags if i]
        categories = [item[0] for item in main_tags]
        categories = list(set(categories))
        return categories

    # Handling of physics sub-categories'
    def physics_tags(tags):
        physics_categories = ['astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin',
                              'nucl-ex', 'nucl-th', 'physics', 'quant-ph']
        result = ['physics' if item in physics_categories else item for item in tags]
        return list(set(result))

    df['main_categories'] = df['categories'].apply(main_categories)
    df['main_categories'] = df['main_categories'].apply(physics_tags)


#
# Writes dataset statistics to file
#

def output_subcategory_stats(df, output_path, summary_file):
    # compute stats
    max_category_count = max(len(x) for x in df['categories'])
    category_count_to_abstract_count = [0] * max_category_count
    for categories in df['categories']:
        category_count_to_abstract_count[len(categories) - 1] += 1

    # generate bar plot
    category_count_names = [str(x + 1) for x in range(0, max_category_count)]
    y_pos = np.arange(len(category_count_names))
    plt.figure()
    plt.bar(y_pos, category_count_to_abstract_count, align='center', alpha=0.5)
    plt.xticks(y_pos, category_count_names)
    plt.ylabel('Abstract Count')
    plt.xlabel('Subcategory Count')
    plt.title('Abstract Count vs Subcategory Count')
    plt.savefig(os.path.join(output_path, 'subcategory_counts.png'))

    # output to summary file
    summary_file.write("# Subcategory Stats\n")
    summary_file.write("Subcategory Count\tAbstract Count\n")
    for i in range(0, len(category_count_to_abstract_count)):
        summary_file.write("%d\t%d\n" % (i, category_count_to_abstract_count[i]))
    summary_file.write("\n\n")


def output_main_category_stats(df, output_path, summary_file):
    # compute stats
    max_category_count = max(len(x) for x in df['main_categories'])
    category_count_to_abstract_count = [0] * max_category_count
    for categories in df['main_categories']:
        category_count_to_abstract_count[len(categories) - 1] += 1

    # generate bar plot
    category_count_names = [str(x + 1) for x in range(0, max_category_count)]
    y_pos = np.arange(len(category_count_names))
    plt.figure()
    plt.bar(y_pos, category_count_to_abstract_count, align='center', alpha=0.5)
    plt.xticks(y_pos, category_count_names)
    plt.ylabel('Abstract Count')
    plt.xlabel('Main Category Count')
    plt.title('Abstract Count vs Main Category Count')
    plt.savefig(os.path.join(output_path, 'maincategory_counts.png'))

    # output to summary file
    summary_file.write("# Main Category Stats\n")
    summary_file.write("Main Category Count\tAbstract Count\n")
    for i in range(0, len(category_count_to_abstract_count)):
        summary_file.write("%d\t%d\n" % (i, category_count_to_abstract_count[i]))
    summary_file.write("\n\n")


def output_main_category_tuple_stats(df, output_path, summary_file):
    # compute stats
    category_tuple_to_count = {}
    for categories in df['main_categories']:
        category_tuple = "+".join(sorted(categories))
        if not category_tuple in category_tuple_to_count:
            category_tuple_to_count[category_tuple] = 0
        category_tuple_to_count[category_tuple] += 1

    # generate bar plot
    sorted_items = sorted(category_tuple_to_count.items(), key=operator.itemgetter(1))
    category_tuples = []
    counts = []
    for category_tuple, count in sorted_items:
        category_tuples.append(category_tuple)
        counts.append(count)

    y_pos = np.arange(len(category_tuples))
    plt.figure(figsize=(15, 20))
    plt.barh(y_pos, counts, align='center', alpha=0.5)
    plt.yticks(y_pos, category_tuples)
    plt.xlabel('Abstract Count')
    plt.ylabel('Category Combination Count')
    plt.title('Abstract Count vs Category Combination')
    plt.savefig(os.path.join(output_path, 'category_combination_counts.png'), dpi=300)

    # output to summary file
    summary_file.write("# Category Combination Stats\n")
    summary_file.write("Total Category Combinations: %d\n" % len(category_tuple_to_count))
    summary_file.write("Category Combination\tAbstract Count\n")
    for category_tuple, count in reversed(sorted_items):
        summary_file.write("%s\t%d\n" % (category_tuple, count))
    summary_file.write("\n\n")


def output_dataset_stats(df, output_path):
    with open(os.path.join(output_path, "dataset_summary.txt"), 'w') as summary_file:
        output_subcategory_stats(df, output_path, summary_file)
        output_main_category_stats(df, output_path, summary_file)
        output_main_category_tuple_stats(df, output_path, summary_file)


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

    df = pd.read_csv(args.input, nrows=args.max_samples)
    df.head()

    # split categories string into list of categories
    df['categories'] = df['categories'].str.split()

    generate_main_categories(df)
    output_dataset_stats(df, args.output)

    #what are main categories?
    unique_categories = set()
    for n in df.main_categories:
        unique_categories.update(n)
    category_to_id = dict([(j,i) for i, j in enumerate(sorted(unique_categories))])

    def OneHotEncoder(tags):
        vec = [0] * len(category_to_id)
        for tag in tags:
            vec[category_to_id[tag]]=1
        return vec

    y_df = df['main_categories'].apply(OneHotEncoder)
    y_df = pd.DataFrame(y_df.values.tolist(), columns=range(0, len(category_to_id)))

    print("processing abstract text")
    #ABSTRACT TEXT PROCESSING
    stopWordList = stopwords.words('english')

    lemma = WordNetLemmatizer()
    token = ToktokTokenizer()

    def removeSpecialChars(text):
        str_chars = '`-=~@#$%^&*()_+[!{;”:\’><.,/?”}]'
        for w in text:
            if w in str_chars:
                text = text.replace(w,'')
        return text

    def lemitizeWords(text):
        words = token.tokenize(text)
        listLemma =[]
        for w in words:
            x = lemma.lemmatize(w,'v')
            listLemma.append(x)
        text = " ".join(listLemma)
        return text

    def stopWordsRemove(text):
        wordList = [x.strip() for x in token.tokenize(text)]
        removedList = [x for x in wordList if not x in stopWordList]
        text =" ".join(removedList)
        return text

    def preprocessingText(text):
        text = removeSpecialChars(text)
        text = lemitizeWords(text)
        text = stopWordsRemove(text)
        return text

    df['abstract'] = df['abstract'].map(lambda x: preprocessingText(x))

    #wordcloud representation
    import sys

    totalText = ''
    for x in df.abstract:
        totalText = totalText + ' ' + x


    wc = WordCloud(background_color='black', max_font_size=50).generate(totalText)
    plt.figure(figsize=(16, 12))
    plt.imshow(wc, interpolation="bilinear")    # mapping integer feature names to original token string
    plt.savefig(os.path.join(args.output, 'word_cloud.png'))


    x1 = df['title'].values
    x2 = df['abstract'].values
    y = y_df.values
    x = np.vstack((x1, x2))
    x = x.T

    ##MAIN CATEGORY CLASSIFICATION
    #Binary relevance

    # split a training set and a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print("%d documents in training set" %len(x_train))
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

    for i in range(0, 10):
        # selecting the best k features from the data set
        feature_ratio = float(i + 1) / 10.0
        feature_count = int(((i + 1) * len(feature_names)) / 10)
        print("Extracting %.1f%% best features by a chi-squared test" % feature_ratio)
        x_train_sel, x_test_sel, feature_names_sel = chi_square_feature_selection(x_train, y_train, x_test, feature_names, feature_count)

        print("Training Binary Relevance classifier")

        if args.binary_relevance_naive_bayes:
            classifier = BinaryRelevance(GaussianNB())
        elif args.binary_relevance_logistic_regression:
            classifier = BinaryRelevance(LogisticRegression())
        else:
            print("ERROR: specify classification model")
            exit()

        classifier.fit(x_train_sel, y_train)

        predictions = classifier.predict(x_test_sel.astype(float))
        predictions = predictions.todense()

        # conf_mat = metrics.confusion_matrix(x_test[:,1], predictions[:,1])

        accuracy_score = metrics.accuracy_score(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)

        # output results to file for this feature count experiment
        output_results(sys.stdout, accuracy_score, classification_report, category_to_id)
        with open(os.path.join(args.output, "results_%d.txt" % int(100 * feature_ratio)), 'w') as results_file:
            output_results(results_file, accuracy_score, classification_report, category_to_id)
        print()

        # update best results if needed
        if (accuracy_score > best_accuracy_score):
            best_accuracy_score = accuracy_score
            best_feature_ratio = feature_ratio
            best_feature_count = feature_count
            best_classification_report = classification_report

    print()
    print()
    print("=== summary ===")
    output_summary(sys.stdout, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id)
    with open(os.path.join(args.output, "summary.txt"), 'w') as summary_file:
        output_summary(summary_file, best_feature_count, best_feature_ratio, best_accuracy_score, best_classification_report, category_to_id)
