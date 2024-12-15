from sklearn.metrics import roc_auc_score
from dkn import DKN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.train_aucs = []
        self.test_aucs = []
        self.losses = []

    def add_metrics(self, epoch, train_auc, test_auc, avg_loss):
        self.epochs.append(epoch)
        self.train_aucs.append(train_auc)
        self.test_aucs.append(test_auc)
        self.losses.append(avg_loss)

    def save_plots(self, save_dir='training_plots'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Plot AUC curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_aucs, 'b-', label='Training AUC')
        plt.plot(self.epochs, self.test_aucs, 'g-', label='Test AUC')
        plt.title('Model Performance: AUC Scores')
        plt.xlabel('Epoch')
        plt.ylabel('AUC Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'auc_curves.png'))
        plt.close()

        # Plot Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, 'r-', label='Average Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        plt.close()

        print("\nTraining Summary:")
        print("-" * 50)
        print(f"Initial Training AUC: {self.train_aucs[0]:.4f}")
        print(f"Final Training AUC:   {self.train_aucs[-1]:.4f}")
        print(f"Initial Test AUC:     {self.test_aucs[0]:.4f}")
        print(f"Final Test AUC:       {self.test_aucs[-1]:.4f}")
        print(f"Initial Loss:         {self.losses[0]:.4f}")
        print(f"Final Loss:           {self.losses[-1]:.4f}")
        print("-" * 50)
        print(f"\nPlots saved to: {save_dir}/")


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

    # Initialize visualizer
    visualizer = TrainingVisualizer()

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
        avg_loss = total_loss / len(start_list)

        print(f'epoch {epoch} train_auc: {train_auc:.4f} test_auc: {test_auc:.4f} '
              f'avg_loss: {avg_loss:.4f}')

        # Add metrics to visualizer
        visualizer.add_metrics(epoch, train_auc, test_auc, avg_loss)

    # Generate and save plots after training
    visualizer.save_plots()
