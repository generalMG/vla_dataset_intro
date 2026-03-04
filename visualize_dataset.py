"""
VLA Dataset Visualizer — inspect and visualize collected episodes.

Usage:
    python visualize_dataset.py                          # visualize episode 0
    python visualize_dataset.py --episode 3              # specific episode
    python visualize_dataset.py --episode 0 --save       # save images instead of showing
    python visualize_dataset.py --all --save             # save grid for all episodes
"""

import argparse
import os
import sys

import h5py
import numpy as np

DATASET_DIR = "/mnt/d/mg_ai_research/workspace/whatnot/vla_dataset"


def load_episode(episode_id):
    path = os.path.join(DATASET_DIR, f"episode_{episode_id:06d}.hdf5")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    return h5py.File(path, "r"), path


def print_summary(f, path):
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(path)}")
    print(f"{'='*60}")
    print(f"  Task:        {f.attrs.get('task_type', 'N/A')}")
    print(f"  Target:      {f.attrs.get('target_object', 'N/A')}")
    print(f"  Instruction: {f.attrs.get('language_instruction', 'N/A')}")
    print(f"  Frames:      {f.attrs.get('num_frames', 'N/A')}")
    print(f"  Success:     {f.attrs.get('success', 'N/A')}")
    print()

    # Data ranges
    wrist = f["observation/images/wrist_rgb"]
    overhead = f["observation/images/overhead_rgb"]
    state = f["observation/state"]
    action = f["action"]
    ee = f["observation/ee_pose"]
    grip = f["observation/gripper_width"]

    print(f"  Wrist RGB:    shape={wrist.shape}, range=[{wrist[()].min()}, {wrist[()].max()}]")
    print(f"  Overhead RGB: shape={overhead.shape}, range=[{overhead[()].min()}, {overhead[()].max()}]")
    print(f"  State:        shape={state.shape}, range=[{state[()].min():.4f}, {state[()].max():.4f}]")
    print(f"  Action:       shape={action.shape}, range=[{action[()].min():.4f}, {action[()].max():.4f}]")
    print(f"  EE Pose:      shape={ee.shape}")
    print(f"  Gripper:      shape={grip.shape}, range=[{grip[()].min():.4f}, {grip[()].max():.4f}]")

    # Depth stats
    wd = f["observation/depth/wrist_depth"][()]
    od = f["observation/depth/overhead_depth"][()]
    print(f"  Wrist Depth:  shape={wd.shape}, range=[{wd.min():.3f}, {wd.max():.3f}]")
    print(f"  Overhead Dep: shape={od.shape}, range=[{od.min():.3f}, {od.max():.3f}]")
    print()


def visualize_episode(episode_id, save=False, output_dir=None):
    """Create a visualization grid for one episode."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    f, path = load_episode(episode_id)
    print_summary(f, path)

    n_frames = f.attrs.get("num_frames", f["action"].shape[0])
    instruction = f.attrs.get("language_instruction", "N/A")
    if isinstance(instruction, bytes):
        instruction = instruction.decode()

    wrist_rgb = f["observation/images/wrist_rgb"][()]
    overhead_rgb = f["observation/images/overhead_rgb"][()]
    wrist_depth = f["observation/depth/wrist_depth"][()]
    overhead_depth = f["observation/depth/overhead_depth"][()]
    joint_pos = f["observation/state"][()]
    joint_vel = f["observation/joint_velocities"][()]
    ee_pose = f["observation/ee_pose"][()]
    gripper = f["observation/gripper_width"][()]
    timestamps = f["observation/timestamps"][()]

    # Pick evenly spaced frames to show
    n_show = min(8, n_frames)
    indices = np.linspace(0, n_frames - 1, n_show, dtype=int)

    # --- Figure 1: Image grid ---
    fig = plt.figure(figsize=(n_show * 3, 12))
    fig.suptitle(f"Episode {episode_id}: \"{instruction}\"\n"
                 f"Frames: {n_frames} | Success: {f.attrs.get('success', 'N/A')}",
                 fontsize=13, fontweight="bold")

    gs = GridSpec(4, n_show, figure=fig, hspace=0.3, wspace=0.05)

    for col, idx in enumerate(indices):
        # Row 0: Wrist RGB
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(wrist_rgb[idx])
        ax.set_title(f"t={idx}", fontsize=9)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Wrist RGB", fontsize=10)

        # Row 1: Overhead RGB
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(overhead_rgb[idx])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Overhead RGB", fontsize=10)

        # Row 2: Wrist Depth
        ax = fig.add_subplot(gs[2, col])
        d = wrist_depth[idx]
        valid = d[d < 1e4]  # filter inf/large values
        vmax = valid.max() if len(valid) > 0 else 1.0
        ax.imshow(np.clip(d, 0, vmax), cmap="viridis")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Wrist Depth", fontsize=10)

        # Row 3: Overhead Depth
        ax = fig.add_subplot(gs[3, col])
        d = overhead_depth[idx]
        valid = d[d < 1e4]
        vmax = valid.max() if len(valid) > 0 else 1.0
        ax.imshow(np.clip(d, 0, vmax), cmap="viridis")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Overhead Depth", fontsize=10)

    if save:
        out_dir = output_dir or os.path.join(DATASET_DIR, "visualizations")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"episode_{episode_id:06d}_images.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()

    # --- Figure 2: Joint trajectories ---
    fig2, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig2.suptitle(f"Episode {episode_id} — Robot State Trajectories", fontsize=13, fontweight="bold")

    t = timestamps if len(timestamps) == n_frames else np.arange(n_frames)

    # Joint positions
    ax = axes[0]
    joint_labels = [f"J{i+1}" for i in range(7)] + ["F1", "F2"]
    for j in range(min(9, joint_pos.shape[1])):
        ax.plot(t, joint_pos[:, j], label=joint_labels[j], linewidth=1)
    ax.set_ylabel("Joint Position (rad)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # End-effector position
    ax = axes[1]
    ee_labels = ["X", "Y", "Z"]
    for j in range(3):
        ax.plot(t, ee_pose[:, j], label=f"EE {ee_labels[j]}", linewidth=1.5)
    ax.set_ylabel("EE Position (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Gripper width + joint velocity magnitude
    ax = axes[2]
    vel_mag = np.linalg.norm(joint_vel[:, :7], axis=1)
    ax.plot(t, gripper, label="Gripper Width", color="tab:blue", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(t, vel_mag, label="Joint Vel |mag|", color="tab:orange", linewidth=1, alpha=0.7)
    ax.set_ylabel("Gripper Width (m)", color="tab:blue")
    ax2.set_ylabel("Velocity Magnitude", color="tab:orange")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if save:
        out_path = os.path.join(out_dir, f"episode_{episode_id:06d}_trajectories.png")
        fig2.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig2)
    else:
        plt.show()

    f.close()


def main():
    parser = argparse.ArgumentParser(description="VLA Dataset Visualizer")
    parser.add_argument("--episode", type=int, default=0, help="Episode ID to visualize")
    parser.add_argument("--all", action="store_true", help="Visualize all episodes")
    parser.add_argument("--save", action="store_true", help="Save images to disk instead of displaying")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for saved images")
    parser.add_argument("--summary-only", action="store_true", help="Print summary without plots")
    args = parser.parse_args()

    if args.summary_only:
        if args.all:
            eps = sorted([
                int(fn.split("_")[1].split(".")[0])
                for fn in os.listdir(DATASET_DIR)
                if fn.startswith("episode_") and fn.endswith(".hdf5")
            ])
            for ep_id in eps:
                f, path = load_episode(ep_id)
                print_summary(f, path)
                f.close()
        else:
            f, path = load_episode(args.episode)
            print_summary(f, path)
            f.close()
        return

    if args.all:
        eps = sorted([
            int(fn.split("_")[1].split(".")[0])
            for fn in os.listdir(DATASET_DIR)
            if fn.startswith("episode_") and fn.endswith(".hdf5")
        ])
        for ep_id in eps:
            visualize_episode(ep_id, save=True, output_dir=args.output_dir)
    else:
        visualize_episode(args.episode, save=args.save, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
