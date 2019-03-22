# -*- coding: utf-8 -*-

import argparse
import os
import operator
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV dataset file")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    parser.add_argument("-ms", "--max-samples", type=int, help="Maximum number of samples to use for processing (0 - use entire dataset).")
    return parser.parse_args()

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

    df['main_categories'] = df['subcategories'].apply(main_categories)
    df['main_categories'] = df['main_categories'].apply(physics_tags)


#
# Writes dataset statistics to file
#

def output_subcategory_stats(df, output_path, summary_file):
    # compute stats
    max_category_count = max(len(x) for x in df['subcategories'])
    category_count_to_abstract_count = [0] * max_category_count
    for categories in df['subcategories']:
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

def randomize_dataset(df):
    df_new = df.sample(frac=1).reset_index(drop=True)
    return df_new


#
# text preprocessing
#   - generate main categories from subcategories
#   - output set statistics
#   - randomize data set
#
#
#
#
if __name__ == "__main__":
    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df = pd.read_csv(args.input, nrows=args.max_samples)
    df.head()

    # split subcategories string into list of subcategories
    df['subcategories'] = df['subcategories'].str.split()

    generate_main_categories(df)
    output_dataset_stats(df, args.output)

    #what are main categories?
    unique_categories = set()
    for n in df.main_categories:
        unique_categories.update(n)
    category_to_id = dict([(j,i) for i, j in enumerate(sorted(unique_categories))])

    def one_hot_encoder(tags):
        vec = [0] * len(category_to_id)
        for tag in tags:
            vec[category_to_id[tag]]=1
        return vec

    y_df = df['main_categories'].apply(one_hot_encoder)
    y_df = pd.DataFrame(y_df.values.tolist(), columns=range(0, len(category_to_id)))

    print("processing abstract text")
    #ABSTRACT TEXT PROCESSING
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

    #wordcloud representation
    import sys

    totalText = ''
    for x in df.abstract:
        totalText = totalText + ' ' + x


    wc = WordCloud(background_color='black', max_font_size=50).generate(totalText)
    plt.figure(figsize=(16, 12))
    plt.imshow(wc, interpolation="bilinear")    # mapping integer feature names to original token string
    plt.savefig(os.path.join(args.output, 'word_cloud.png'))


    df_rand = randomize_dataset(df)
    df_rand.to_csv(os.path.join(args.output,os.path.splitext(args.input)[0] + '_processed.csv'))