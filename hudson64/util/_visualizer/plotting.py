import matplotlib.pyplot as plt
import os

class LossPlotter:
    """
    Simple loss plotter that:
    - keeps track of losses per epoch
    - plots them live
    - can save the figure to a PNG
    """

    def __init__(self, title="Training Loss", save_path="loss_curve.png"):
        self.losses = []
        self.title = title
        self.save_path = save_path

        # create figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self.title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")

        plt.ion()   # interactive mode ON
        plt.show()

    def add_loss(self, loss):
        """Record a new epoch loss and update plot."""
        self.losses.append(loss)

        self.ax.clear()
        self.ax.plot(self.losses, marker="o")
        self.ax.set_title(self.title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")

        # redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # save image
        self.fig.savefig(self.save_path)

        print(f"Saved loss plot → {self.save_path}")

    def finalize(self):
        """Turn interactive mode off and save final plot."""
        plt.ioff()
        self.fig.savefig(self.save_path)
        print(f"Final loss curve saved to {self.save_path}")
        plt.show()
