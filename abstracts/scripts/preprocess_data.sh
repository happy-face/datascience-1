#!/bin/bash

#python3 preprocess_data.py -i raw_data/test_data_jan_2018.csv -o data/test
#python3 preprocess_data.py -i raw_data/train_data_feb_march_2018.csv -o data/train
python3 preprocess_data.py -i raw_data/train_data_jan_june_2018.csv -o data/train_jan_june_2018 --force
