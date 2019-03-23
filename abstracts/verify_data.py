import pandas as pd

def count_values(values):
    value2count = {}
    for line in values:
        if not line in value2count:
            value2count[line] = 0
        value2count[line] += 1
    return value2count

if __name__ == "__main__":
    df_train = pd.read_csv("data/train/train_data_feb_march_2018.csv_processed.csv")
    df_train.head()

    df_train = df_train.drop_duplicates('abstract')
    df_train = df_train.drop_duplicates('title')

    abs2count = count_values(df_train.abstract)
    abs_count_histogram = count_values(abs2count.values())
    print "abs histogram: " + str(abs_count_histogram)

    tit2count = count_values(df_train.title)
    tit_count_histogram = count_values(tit2count.values())
    print "tit histogram: " + str(tit_count_histogram)
