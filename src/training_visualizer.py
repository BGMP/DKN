import matplotlib.pyplot as plt
import os


class TrainingVisualizer:
    """Handles visualization of training metrics for DKN model."""

    def __init__(self):
        self.epochs = []
        self.train_aucs = []
        self.test_aucs = []
        self.losses = []

    def add_epoch_metrics(self, epoch, train_auc, test_auc, avg_loss):
        """Store metrics for each epoch."""
        self.epochs.append(epoch)
        self.train_aucs.append(train_auc)
        self.test_aucs.append(test_auc)
        self.losses.append(avg_loss)

    def plot_metrics(self, save_dir='./training_plots'):
        """Generate and save visualization plots."""
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Set style for better visibility
        plt.style.use('seaborn')

        # Plot AUC curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_aucs, 'b-', label='Training AUC', linewidth=2)
        plt.plot(self.epochs, self.test_aucs, 'g-', label='Test AUC', linewidth=2)
        plt.title('DKN Model Performance: AUC Scores During Training')
        plt.xlabel('Epoch')
        plt.ylabel('AUC Score')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'auc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, 'r-', label='Average Loss', linewidth=2)
        plt.title('DKN Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Print summary statistics
        print("\nTraining Summary:")
        print("-" * 50)
        print(f"Initial Training AUC: {self.train_aucs[0]:.4f}")
        print(f"Final Training AUC:   {self.train_aucs[-1]:.4f}")
        print(f"Best Training AUC:    {max(self.train_aucs):.4f}")
        print(f"Initial Test AUC:     {self.test_aucs[0]:.4f}")
        print(f"Final Test AUC:       {self.test_aucs[-1]:.4f}")
        print(f"Best Test AUC:        {max(self.test_aucs):.4f}")
        print(f"Initial Loss:         {self.losses[0]:.4f}")
        print(f"Final Loss:           {self.losses[-1]:.4f}")
        print("-" * 50)
        print(f"\nVisualization plots have been saved to: {save_dir}/")
