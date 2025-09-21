import matplotlib.pyplot as plt

def plot_loss(history, title: str = "Model Loss Over Epochs"):
    """
    Plot training and validation loss from Keras History object.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="Train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
