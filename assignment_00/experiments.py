"""
Join the competition and get the preprocessed dataset. Please make submissions
as many as you can, which will be evaluated with AUC by Kaggle and revealed on
leaderboard. There are several benchmarks on the learderboard. Each of them
was predicted with different model. We expect you to get at least 0.82152 (
Santino Corleone) in this assignment. After you are done, send back the report
describing how you build your prediction model, the source code, and MD5 of
your best submissions along with your Kaggle account for confirmation.

As our intention is to evaluate your skill, not taking advantage of you, feel
free to refuse providing the source code if you beat the second benchmark on
the leaderboard (Vito Corleone, 0.89064).

Things you can do:
    * Implement a model and evaluate it.
    * Implement a model from scratch (e.g. do deep learning with numpy).
    * Extract and try different features from raw data.

Note:
    * We expect you to get at least 0.82152.
    * We have prepare the code to explain the pre-processed dataset, feel free
      to change/replace it.
    * If it's done in python, please send *.py instead *.ipynb.
    * Although this is not a coding style test, please make the code readable,
      please...
"""
import csv
import numpy as np
import os


def predict(features):
    ""
    ""
    # NOTE: guess with random probabilities. the AUC should be closed to 0.5.
    n, v = features.shape

    return np.random.random((n, 28))


def write_result(name, predictions):
    """
    """
    if predictions is None:
        raise Exception('need predictions')

    predictions = predictions.flatten()

    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    path = os.path.join('./results/', name)

    with open(path, 'wt', encoding='utf-8', newline='') as csv_target_file:
        target_writer = csv.writer(csv_target_file, lineterminator='\n')

        header = [
            'user_id',
            'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
            'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
            'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
            'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
            'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
            'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
            'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27',
        ]

        target_writer.writerow(header)

        for i in range(0, len(predictions), 28):
            # NOTE: 57159 is the offset of user ids
            userid = [57159 + i // 28]
            labels = predictions[i:i+28].tolist()

            target_writer.writerow(userid + labels)


# NOTE: load the data from the npz
dataset = np.load('./datasets/v0_eigens.npz')

# NOTE: calculate the size of training set and validation set
#       all pre-processed features are inside 'train_eigens'
train_data_size = dataset['train_eigens'].shape[0]
valid_data_size = train_data_size // 5
train_data_size = train_data_size - valid_data_size

# NOTE: split dataset
train_data = dataset['train_eigens'][:train_data_size]
valid_data = dataset['train_eigens'][train_data_size:]

# NOTE: a 896d feature vector for each user, the 28d vector in the end are
#       labels
#       896 = 32 (weeks) x 7 (days a week) x 4 (segments a day)
train_eigens = train_data[:, :-28].reshape(-1, 896)
train_labels = train_data[:, -28:]

valid_eigens = valid_data[:, :-28].reshape(-1, 896)
valid_labels = valid_data[:, -28:]

# NOTE: read features of test set
test_eigens = dataset['issue_eigens'][:, :-28].reshape(-1, 896)

# NOTE: check the shape of the prepared dataset
print('train_eigens.shape = {}'.format(train_eigens.shape))
print('train_labels.shape = {}'.format(train_labels.shape))
print('valid_eigens.shape = {}'.format(valid_eigens.shape))
print('valid_labels.shape = {}'.format(valid_labels.shape))

# NOTE: predict and save
test_guesss = predict(test_eigens)

write_result('dnn.csv', test_guesss)
