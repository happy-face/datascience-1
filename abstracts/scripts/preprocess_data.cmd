REM preprocess test and training set
python preprocess_data.py -i raw_data\test_data_jan_2018.csv -o data\test --force
python preprocess_data.py -i raw_data\train_data_feb_march_2018.csv -o data\train --force

REM create mini test/train sets of 100 samples for code testing purposes
python preprocess_data.py -i raw_data\test_data_jan_2018.csv -o data\mini_test --force --max-rows 100
python preprocess_data.py -i raw_data\train_data_feb_march_2018.csv -o data\mini_train --force --max-rows 100

