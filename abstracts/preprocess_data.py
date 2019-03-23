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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV dataset file")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("-mr", "--max-rows", type=int, help="Limit number of rows to read from input CSV file.")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
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
                              'nucl-ex', 'nucl-th', 'physics', 'quant-ph', 'chao-dyn']
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


def supported_categories(categories):
    supported = set(["physics", "stat", "cs", "math", "q-bio", "q-fin", "eess"])
    for cat in categories:
        if cat not in supported:
            return False
    return True

if __name__ == "__main__":
    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df = pd.read_csv(args.input, nrows=args.max_rows)
    df.head()

    # remove duplicates
    df = df.drop_duplicates('abstract')
    df = df.drop_duplicates('title')

    # split subcategories string into list of subcategories
    df['subcategories'] = df['subcategories'].str.split()
    generate_main_categories(df)

    # keep only supported categories
    df = df[df.apply(lambda x: supported_categories(x['main_categories']), axis=1)]

    output_dataset_stats(df, args.output)

    totalText = ''
    for x in df.abstract:
        totalText = totalText + ' ' + x


    wc = WordCloud(background_color='black', max_font_size=50).generate(totalText)
    plt.figure(figsize=(16, 12))
    plt.imshow(wc, interpolation="bilinear")    # mapping integer feature names to original token string
    plt.savefig(os.path.join(args.output, 'word_cloud.png'))


    df = randomize_dataset(df)

    out_file_name = os.path.splitext(os.path.basename(args.input))[0]
    df.to_csv(os.path.join(args.output, out_file_name + '_processed.csv'))


    print()
    print("Category statistics:")
    cat2count = {}
    for el in df.main_categories:
        for cat in el:
            if cat not in cat2count:
                cat2count[cat] = 0
            cat2count[cat] += 1

    total = sum(cat2count.values())
    for cat, count in cat2count.items():
        print("%s\t%d\t%.1f%%" % (cat, count, 100 * float(count) / total))
