"""
Join the competition and get both raw and pre-processed datasets. Run the
following script to make predictions with pre-processed dataset. Submit the
result to Kaggle and get the score. The score should be around 0.85. After
that, write your own script to process the raw dataset. Your goal is to create
a new pre-processed dataset just like the one we provided. After you are done,
send back the source code, and MD5 of your final submission along with your
Kaggle account for confirmation.

Pre-processed dataset:
-   train.csv for training.
-   test.csv for testing.
-   Each row represents one user.
-   Each column represents one time segment
    (924 = 4 (segments a day) x 7 (days a week) x 33 (weeks)).
-   The value 0/1 means if the user (row) played a content during that time
    segment.
-   Use the first 896 columns (row[:896]) to infer the last 28 columns
    (row[-28:]) for each user/row.

Note:
-   It's ok if your pre-processed dataset is slightly different from ours.
    But the prediction base on your preprocessed dataset should be at least
    0.82152.
-   Building a better model to get better scores is a plus, but it's not
    necessary. We will focus on the data processing skills here.
-   If it's done in python, please send *.py (no *.ipynb).
-   Although this is not a coding style test, please make the code readable,
    please...
"""
import gzip

import numpy as np
import sklearn.linear_model


def read_dataset(training_dataset_path, testing_dataset_path):
    """
    Read pre-processed datasets for training and testing.
    """
    training_dataset = np.genfromtxt(training_dataset_path, delimiter=',')
    testing_dataset = np.genfromtxt(testing_dataset_path, delimiter=',')

    # NOTE: Use the first 896 columns to infer the last 28 columns.
    training_features = training_dataset[:, :896]
    training_labels = training_dataset[:, 896:]
    testing_features = testing_dataset[:, :896]

    return training_features, training_labels, testing_features


def train_and_predict(training_features, training_labels, testing_features):
    """
    Train the model and make predictions.
    """
    model = sklearn.linear_model.LinearRegression()

    model.fit(training_features, training_labels)

    testing_results = model.predict(testing_features)

    return testing_results


def write_result(path, predictions):
    """
    """
    if predictions is None:
        raise ValueError('need predictions')

    # NOTE: user_id of the testing dataset starts from 57159.
    user_ids = np.arange(57159, 57159 + 37092)

    # NOTE: Concatenate user IDs for kaggle.
    user_ids = np.expand_dims(user_ids, axis=1)

    results = np.concatenate([user_ids, predictions], axis=-1)

    header = ['user_id'] + ['time_slot_{}'.format(idx) for idx in range(28)]

    header = ','.join(header)

    fmt = ['%d'] + ['%f' for idx in range(28)]

    with gzip.open(path, 'wt', encoding='utf-8', newline='') as gz_file:
        np.savetxt(
            gz_file,
            results,
            fmt=fmt,
            delimiter=',',
            header=header,
            comments='')


if __name__ == '__main__':
    training_features, training_labels, testing_features = \
        read_dataset('./train.csv', './test.csv')

    predictions = train_and_predict(
        training_features, training_labels, testing_features)

    write_result('results.gz', predictions)
