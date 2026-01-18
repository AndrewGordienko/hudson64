import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from hudson64.core._network.model import AlphaZeroNet

DEVICE = "cpu"


def _load_shard(path):
    data = torch.load(path, map_location="cpu")
    X = data["inputs"]
    Pi = data["policy"]
    V = data["value"]
    return TensorDataset(X, Pi, V)


def _train_one_epoch(model, optimizer, datasets, batch_size):
    model.train()
    total_loss = 0.0
    total_items = 0

    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for xb, pi_b, v_b in tqdm(loader, desc="Training batches", leave=False):
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
            total_items += xb.size(0)

    return total_loss / max(1, total_items)


# =====================================================================
# TRAINING FUNCTION FOR ALPHAZERO LOOP
# =====================================================================

def train_on_replay_buffers(
    replay_paths,
    model_path_in=None,
    model_path_out="policy_value_best.pt",
    epochs=1,
    batch_size=128,
    lr=1e-4,
):
    """
    replay_paths: list of replay buffer PT files
    model_path_in: load previous best model (None = initialize fresh)
    model_path_out: save updated model
    """

    print("\n=== TRAINING ON REPLAY BUFFERS ===")
    print(f"Replay buffers: {len(replay_paths)} files")

    # Load datasets
    datasets = []
    for path in replay_paths:
        print("  loading", path)
        datasets.append(_load_shard(path))

    # Init model
    model = AlphaZeroNet().to(DEVICE)

    if model_path_in is not None and os.path.exists(model_path_in):
        print(f"Loading previous model: {model_path_in}")
        model.load_state_dict(torch.load(model_path_in, map_location="cpu"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for ep in range(1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        avg_loss = _train_one_epoch(model, optimizer, datasets, batch_size)
        print(f"  Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), model_path_out)
    print(f"Saved updated model → {model_path_out}")

    return model_path_out
