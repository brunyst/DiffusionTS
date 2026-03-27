from utils.imports_statiques import *


def plot_avg_loss_curves(train_losses, val_losses, title='Training & Validation Loss', figsize=(14, 6), save_dir=None, plot_path="loss_curves"):
    """
    Plots the training and validation loss curves over epochs.

    :param train_losses: [array]; avg training loss values at each epoch.
    :param val_losses: [array]; avg validation loss values at each epoch.
    :param title: [str]; title of the plot.
    :param figsize: [tuple]; size of the figure in inches.
    :param save_dir: [str|None]; directory to save the figure. If None, figure is not saved.
    :param plot_path: [str]; base filename (stem used).
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss value')
    plt.title(title)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    full_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(plot_path))[0]
        full_path = os.path.join(save_dir, f"{stem}_loss_curves.png")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

    plt.close('all')
    return full_path