import torch
import os
import shutil

from hudson64.core._rl.selfplay import selfplay_many_games
from hudson64.core._rl.train_network import train_on_replay_buffers
from hudson64.core._rl.evaluate_models import evaluate_models 


def main():
    ITERATIONS      = 20
    NUM_GAMES       = 20     # self-play per iteration
    SIMS_PER_MOVE   = 100     # increase over time
    REPLAY_WINDOW   = 20

    OLD_MODEL = "policy_value_final.pt"

    os.makedirs("replay_buffers", exist_ok=True)

    replay_files = []

    for it in range(1, ITERATIONS + 1):
        print(f"\n========== ITERATION {it} ==========\n")

        # -----------------------------------------
        # 1) SELF-PLAY -> Replay Buffer
        # -----------------------------------------
        buffer_path = f"replay_buffers/selfplay_buffer_{it}.pt"
        print(f"[Self-play] → {buffer_path}")

        selfplay_many_games(
            model_path=OLD_MODEL,
            out_path=buffer_path,
            games=NUM_GAMES,
            simulations=SIMS_PER_MOVE,
        )

        replay_files.append(buffer_path)
        if len(replay_files) > REPLAY_WINDOW:
            os.remove(replay_files.pop(0))

        # -----------------------------------------
        # 2) TRAIN ON REPLAY BUFFERS
        # -----------------------------------------
        candidate_path = "candidate.pt"
        print("\n[Training] Training candidate model...\n")

        train_on_replay_buffers(
            replay_paths=replay_files,
            model_path_in=OLD_MODEL,
            model_path_out=candidate_path,
            epochs=10,
            batch_size=256,
            lr=1e-4,
        )

        # -----------------------------------------
        # 3) EVALUATE: NEW vs OLD
        # -----------------------------------------
        print("\n[Evaluation] Arena match old vs new...")

        win_rate = evaluate_models(
            old_path=OLD_MODEL,
            new_path=candidate_path,
            games=20,          # increase later
            sims=100,            # search used during evaluation
        )

        print(f"New model win rate = {win_rate:.3f}")

        # -----------------------------------------
        # 4) ACCEPT / REJECT NEW MODEL
        # -----------------------------------------
        if win_rate >= 0.55:
            print("→ New model accepted!\n")
            shutil.copy(candidate_path, OLD_MODEL)
        else:
            print("→ New model rejected, keeping old.\n")


if __name__ == "__main__":
    main()
