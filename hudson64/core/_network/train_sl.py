import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from hudson64.core._network.model import AlphaZeroNet
from hudson64.util._visualizer.plotting import LossPlotter

SHARD_FOLDER = "dataset_chunks"

BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3

EARLY_STOP_MIN_DELTA = 0.001   # stop if improvement < 0.001
EARLY_STOP_PATIENCE = 3        # stop after 3 stagnant epochs

DEVICE = ("cpu")

def load_shard(path):
    print(f"\n[Loading shard] {path}")
    data = torch.load(path, map_location="cpu")

    X = data["inputs"]    # [N, C, 8, 8]
    Pi = data["policy"]   # [N]
    V = data["value"]     # [N, 1]

    print(f"  → Shapes: X={X.shape}, Pi={Pi.shape}, V={V.shape}")
    return TensorDataset(X, Pi, V)


def train_on_shard(model, optimizer, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    model.train()
    total_loss = 0.0

    for xb, pi_b, v_b in tqdm(loader, desc="Shard Batches", leave=False):
        xb = xb.to(DEVICE)
        pi_b = pi_b.to(DEVICE)
        v_b = v_b.to(DEVICE)

        policy_logits, value_pred = model(xb)

        policy_loss = F.cross_entropy(policy_logits, pi_b)
        value_loss = F.mse_loss(value_pred, v_b)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(dataset)


def main():

    print(f"Using device: {DEVICE}")

    shard_paths = sorted(
        os.path.join(SHARD_FOLDER, f)
        for f in os.listdir(SHARD_FOLDER)
        if f.startswith("dataset_sl_chunk_") and f.endswith(".pt")
    )

    print("\nShards found:")
    for p in shard_paths:
        print("  ", p)

    model = AlphaZeroNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    plotter = LossPlotter(title="SL Training Loss", save_path="sl_loss.png")

    best_loss = float("inf")
    stagnant_epochs = 0

    print("\n=== Starting Training ===\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n==============================")
        print(f"   EPOCH {epoch} / {EPOCHS}")
        print(f"==============================")

        epoch_loss = 0.0
        shard_count = 0

        for shard_idx, shard_path in enumerate(shard_paths, 1):
            dataset = load_shard(shard_path)
            avg_loss = train_on_shard(model, optimizer, dataset)

            print(f"[Epoch {epoch}] Shard {shard_idx}/{len(shard_paths)} | Loss: {avg_loss:.5f}")

            epoch_loss += avg_loss
            shard_count += 1

        epoch_loss /= shard_count
        print(f"\n>>> Epoch {epoch} Average Loss: {epoch_loss:.5f}\n")

        # Log to plot
        plotter.add_loss(epoch_loss)

        # Early stopping logic
        if epoch_loss + EARLY_STOP_MIN_DELTA < best_loss:
            best_loss = epoch_loss
            stagnant_epochs = 0
            torch.save(model.state_dict(), "policy_value_best.pt")
            print(f"✓ New best loss — model saved to policy_value_best.pt")
        else:
            stagnant_epochs += 1
            print(f"Loss plateaued ({stagnant_epochs}/{EARLY_STOP_PATIENCE})")

        if stagnant_epochs >= EARLY_STOP_PATIENCE:
            print("\n##### EARLY STOPPING TRIGGERED — Loss plateaued #####\n")
            break

    # Save final model regardless
    torch.save(model.state_dict(), "policy_value_final.pt")
    plotter.finalize()

    print("\nTraining complete.")
    print(f"Best loss: {best_loss:.5f}")
    print("Saved models:")
    print("  - policy_value_best.pt")
    print("  - policy_value_final.pt\n")


if __name__ == "__main__":
    main()
