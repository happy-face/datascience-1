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

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    df = pd.read_csv('TrainingData.csv')
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
    allCategories = []
    for n in df.main_categories:
        allCategories.extend(n)
        allCategories = list(set(allCategories))

    category_to_id = dict([(j,i) for i, j in enumerate(allCategories)])
    def OneHotEncoder(tags):
        vec = [0] * len(allCategories)
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

    totalText = ''
    for x in df.abstract:
        totalText = totalText + ' ' + x


    wc = WordCloud(background_color='black', max_font_size=50).generate(totalText)
    plt.figure(figsize=(16, 12))
    plt.imshow(wc, interpolation="bilinear")

    #frequency of the words
    x = nltk.FreqDist(ToktokTokenizer().tokenize(totalText))
    plt.figure(figsize=(16, 5))
    x.plot(20)


    ##MAIN CATEGORY CLASSIFICATION
    #Binary relevance

    x1 = df['title'].values
    x2 = df['abstract'].values
    y = y_df.values
    x1 = x1[0:5000]
    x2 = x2[0:5000]
    y = y[0:5000]


    cvTitle = CountVectorizer().fit(x1)
    title = pd.DataFrame(cvTitle.transform(x1).todense(), columns=cvTitle.get_feature_names())

    cvAbstract = CountVectorizer().fit(x2)
    abstract = pd.DataFrame(cvAbstract.transform(x2).todense(), columns=cvAbstract.get_feature_names())

    #x = pd.concat([title, abstract], axis=1)


    tfidftitle = TfidfTransformer().fit(title)
    tit = pd.DataFrame(tfidftitle.transform(title).todense())

    tfidfabs = TfidfTransformer().fit(abstract)
    abs = pd.DataFrame(tfidfabs.transform(abstract).todense())

    x = pd.concat([tit,abs], axis=1)


    print("Training Binary Relevance classifier")
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    #selecting only the top k features from the set
    from sklearn.feature_selection import SelectKBest, chi2
    xtrain = SelectKBest(chi2, k=1000).fit_transform(xtrain, ytrain)

    classifier = BinaryRelevance(GaussianNB())
    classifier.fit(xtrain, ytrain)

    predictions = classifier.predict(xtest.astype(float))
    predictions = predictions.todense()


    conf_mat = confusion_matrix(ytest[:,1], predictions[:,1])

    print("Accuracy scores:")
    for i in range(len(allCategories)):
        print(accuracy_score(ytest[:,i], predictions[:,i]))

    print("Full accuracy score:")
    print(accuracy_score(ytest, predictions))