# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

def preprocess_text(df):
    # Any missing values in the data?
    df.isnull().values.any()
    # How balanced are the sets
    fig = plt.figure(figsize=(8,6))
    df.groupby('sentiment').review.count().plot.bar(ylim=0)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-h","--help", help="Help message")
    parser.add_argument("-i", "--input", required=True, help="Input data CSV")
    parser.add_argument("-o", "--output", required=True, help="Output data folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    parser.add_argument("-ms", "--max-samples", type=int, help="Maximum number of samples to use for training (0 - use entire dataset).")

    return parser.parse_args()

# printing statistics and results
def output_results(file, tag, accuracy_score):
    file.write("#\n")
    file.write("# " + tag + "\n")
    file.write("#\n")
    file.write("accuracy scores:" + "\n")
    accuracy_score.to_csv(file, sep='\t')
    file.write("\n")


if __name__ == "__main__":
    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("Reading dataset: %s" % args.input, flush=True)
    df = pd.read_csv(args.input, sep='\t', nrows = args.max_samples)
    df.head()

    preprocess_text(df)

    # Remove all columns except: input - review, output - sentiment

    col = ['sentiment', 'review']
    df = df[col]

    #calculate TF-IDF
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')
    features = tfidf.fit_transform(df.review).toarray()
    print(features.shape)

    #feature selection using chi2 to find the terms most correlated to sentiment
    N=2
    for sentiments in [0,1]:
        print(df.sentiment == sentiments)
        features_chi2 = chi2(features, df.sentiment == sentiments)
        indices = np.argsort(features_chi2[0])
        print(indices)
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print(sentiments)
        print(" . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print(" . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    print(clf.predict(count_vect.transform(["This movie is great."])))
    print(clf.predict(count_vect.transform(["This is the best movie I've ever seen in a long time the fast and the furious"
                                            "is a awsome movie with a great story awsome animation unforgettable characters"
                                            "and songs that will set you free this is a must watched movie for everybody who"
                                            "like games like need for speed underground 2 and spy movies like spy kids I give it a 10/10"])))
    print(clf.predict(count_vect.transform(["They really tried to make the film more than just a racer, but it couldn't keep"
                                            "my attention span for more than 10 minutes at a time, I kept looking for something"
                                            "else to do. The tunnel racing was nicely done, but that's about the only highlight for me. "])))

    print(clf.predict(count_vect.transform(["Since we can't give negative scores this was the only logical score i could give."
                                            "The worst acting ever seen, the worst story ever. Every event turn out to be less"
                                            "logical than a man marruing a jellyfish. All the answer came out from nowhere."
                                            "Why do Brian knew who was the culprit without anyone telling him about it?."
                                            "In the church (the most ridiculous scene in the movie); Why do Brian hang a pistol"
                                            "when the other one was using a shootgun?. Please god, I BEG YOU, Don't ever let another F&F film be made ever again."])))

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0)
    ]

    CV=5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, df.sentiment, cv=CV, pre_dispatch=1)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name','fold_idx','accuracy'])

    accuracy_all_models = cv_df.groupby('model_name')['accuracy'].mean()
    print(accuracy_all_models)
    with open(os.path.join(args.output, "results.txt"), 'w') as results_file:
        output_results(results_file, "TEST SET", accuracy_all_models)

    import seaborn as sns

    sns_plot = sns.boxplot(x='model_name',y='accuracy',data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor='gray',linewidth=2)
    plt.show()
    sns_plot.figure.savefig(os.path.join(args.output, 'model_accuracy_comparison.png'))