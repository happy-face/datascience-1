import pandas as pd

from sklearn.model_selection import train_test_split

def randomize_dataset(df):
    df_new = df.sample(frac=1).reset_index(drop=True)
    return df_new

if __name__ == "__main__":

    df = pd.read_csv("TrainingData.csv")

    df = randomize_dataset(df)

    x_train, x_test = train_test_split(df, test_size=0.25)

    print("%d documents in train set" %len(x_train), flush=True)
    print("%d documents in test set" % len(x_test), flush=True)

    x_train.to_csv("train_data_old_set.csv")
    x_test.to_csv("test_data_old_set.csv")




