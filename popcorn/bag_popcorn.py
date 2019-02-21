import pandas as pd
from io import StringIO
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score


df = pd.read_csv('C:\insight_program\ml_bag_popcorn\labeledTrainData.tsv', sep='\t')
df.head()

# Remove all columns except: input - review, output - sentiment

col = ['sentiment', 'review']
df = df[col]
#dropping columns to prevent memory error
df = df[:-5000]


# Any missing values in the data?
df.isnull().values.any()

# How balanced are the sets
#fig = plt.figure(figsize=(8,6))
#df.groupby('sentiment').review.count().plot.bar(ylim=0)
#plt.show()

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

print(cv_df.groupby('model_name')['accuracy'].mean())

import seaborn as sns

sns.boxplot(x='model_name',y='accuracy',data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor='gray',linewidth=2)
plt.show()