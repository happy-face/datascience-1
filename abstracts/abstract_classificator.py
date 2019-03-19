# -*- coding: utf-8 -*-

import argparse
import os
import operator

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
from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV dataset file")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-ms", "--max-samples", type=int, default=0, help="Maximum number of samples to use for training (0 - use entire dataset).")

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


def output_summary(file, best_feature_count, best_accuracy_score, best_classification_report, category_to_id):
    file.write("best_feature_count = %f.1" % best_feature_count)
    file.write("\n")
    output_results(file, best_accuracy_score, best_classification_report, category_to_id)


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
        print("Output folder %s already exists!" % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df = pd.read_csv(args.input)
    df.head()

    #add number of tags column
    df['categories'] = df['categories'].str.split()
    df['number_of_tags'] = df['categories'].apply(len)
    max_tags = df['number_of_tags'].max()

    #how many abstracts with 1,2,3... tags
    abs_per_tag = [len(df[df['number_of_tags']== (x+1)]) for x in range(max_tags)]

    def main_categories(tags):
        # Function that converts given tags to a list of main categories
        #test = ['cs.it', 'cs.dm', 'math.co', 'math.it']
        main_tags = [i.split(".") for i in tags if i]
        categories = [item[0] for item in main_tags]
        categories = list(set(categories))
        return categories


    physics_categories = ['astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin',
                          'nucl-ex', 'nucl-th', 'physics', 'quant-ph']

    def physics_tags(tags):
        #Function that correct all physics sub-categories to 'physics'
        #tags = ['cs', 'math', 'quant-ph']
        result = ['physics' if item in physics_categories else item for item in tags]
        return list(set(result))


    df['main_categories'] = df['categories'].apply(main_categories)
    df['main_categories'] = df['main_categories'].apply(physics_tags)
    df['number_of_categories'] = df['main_categories'].apply(len)

    max_tags = df['number_of_categories'].max()
    print(max_tags)

    #how many abstracts with 1,2,3... tags
    abs_no_categ = [len(df[df['number_of_categories'] == x+1]) for x in range(max_tags)]

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
    y_df = pd.DataFrame(y_df.values.tolist(), columns=[0, 1, 2, 3, 4, 5, 6, 7])

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

    # Keep only first N features if required
    if args.max_samples != 0:
        x1 = x1[:args.max_samples]
        x2 = x2[:args.max_samples]
        y = y[:args.max_samples]

    import numpy as np
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
    title_train = pd.DataFrame(vec_title.fit_transform(x_train[:,0]).todense(), columns = vec_title.get_feature_names())
    feature_names += get_feature_names("tit-", vec_title)
    title_test = pd.DataFrame(vec_title.transform(x_test[:,0]).todense(), columns = vec_title.get_feature_names())

    # abstract features
    vec_abstract = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1,2))
    abstract_train = pd.DataFrame(vec_abstract.fit_transform(x_train[:,1]).todense(), columns = vec_abstract.get_feature_names())
    feature_names += get_feature_names("abs-", vec_abstract)
    abstract_test = pd.DataFrame(vec_abstract.transform(x_test[:,1]).todense(), columns = vec_abstract.get_feature_names())

    x_train = pd.concat([title_train, abstract_train], axis=1)
    x_test = pd.concat([title_test, abstract_test], axis=1)

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

        classifier = BinaryRelevance(GaussianNB())
        classifier.fit(x_train_sel, y_train)

        predictions = classifier.predict(x_test_sel.astype(float))
        predictions = predictions.todense()

        # conf_mat = metrics.confusion_matrix(x_test[:,1], predictions[:,1])

        accuracy_score = metrics.accuracy_score(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)

        # output results to file for this feature count experiment
        output_results(sys.stdout, accuracy_score, classification_report, category_to_id)
        with open(os.path.join(args.output, "results_10.txt"), 'w') as results_file:
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
    output_summary(sys.stdout, best_feature_count, best_accuracy_score, best_classification_report, category_to_id)
    with open(os.path.join(args.output, "summary.txt"), 'w') as summary_file:
        output_summary(summary_file, best_feature_count, best_accuracy_score, best_classification_report, category_to_id)
