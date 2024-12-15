from sklearn.metrics import roc_auc_score

from dkn import DKN
import numpy as np

import numpy as np
import tensorflow as tf


def get_batch_data(data, start, end):
    return {
        'clicked_words': data.clicked_words[start:end],
        'clicked_entities': data.clicked_entities[start:end],
        'news_words': data.news_words[start:end],
        'news_entities': data.news_entities[start:end],
        'labels': data.labels[start:end]
    }


def train(args, train_data, test_data):
    # Create and compile model
    model = DKN(args)
    model.compile(args)

    for epoch in range(args.n_epochs):
        # Training
        start_list = list(range(0, train_data.size, args.batch_size))
        np.random.shuffle(start_list)
        total_loss = 0

        for start in start_list:
            end = min(start + args.batch_size, train_data.size)
            batch_data = get_batch_data(train_data, start, end)
            loss = model.train_step(batch_data)
            total_loss += loss

        # Evaluation
        train_pred = model.test_step(get_batch_data(train_data, 0, train_data.size))
        test_pred = model.test_step(get_batch_data(test_data, 0, test_data.size))

        train_auc = roc_auc_score(train_data.labels, train_pred)
        test_auc = roc_auc_score(test_data.labels, test_pred)

        print(f'epoch {epoch}    train_auc: {train_auc:.4f}    test_auc: {test_auc:.4f}    '
              f'avg_loss: {total_loss / len(start_list):.4f}')
