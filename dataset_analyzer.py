"""
Dataset Analyzer + Split Balancer for Image Classification
==========================================================
1. Scans train/ and val/ folders
2. Reports class distribution and split statistics
3. Plots horizontal charts (before balancing)
4. Asks what validation percentage you want
5. Randomly moves files class-by-class between train/ and val/
6. Re-plots charts and prints updated analysis

Usage:
    python dataset_analyzer.py --path /path/to/your/dataset

    # Non-interactive balancing:
    python dataset_analyzer.py --path /path/to/dataset --val-percentage 20 --yes

Dependencies:
    pip install matplotlib numpy

Expected folder structure:
    dataset/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── val/
        ├── class1/
        ├── class2/
        └── ...
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# Config
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
IDEAL_SPLIT_MIN = 70
IDEAL_SPLIT_MAX = 85


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if is_image(f)]


def count_images(folder: Path) -> int:
    return len(list_images(folder))


def collect_stats(base_path: Path) -> dict:
    stats = {}
    for split in ("train", "val"):
        split_path = base_path / split
        if not split_path.exists():
            print(f"  [WARNING] Folder not found: {split_path}")
            stats[split] = {}
            continue
        stats[split] = {
            cls_dir.name: count_images(cls_dir)
            for cls_dir in sorted(split_path.iterdir())
            if cls_dir.is_dir()
        }
    return stats


def bar_color(value: int, avg_val: float) -> str:
    ratio = value / avg_val if avg_val else 0
    if ratio >= 0.7:
        return "#378ADD"
    if ratio >= 0.4:
        return "#BA7517"
    return "#E24B4A"


def print_report(stats: dict, all_classes: list[str], val_target_pct: Optional[int] = None):
    train = stats.get("train", {})
    val = stats.get("val", {})

    print("\n" + "=" * 72)
    print("  DATASET ANALYSIS REPORT")
    print("=" * 72)
    print(f"\n  {'Class':<24} {'Train':>8} {'Val':>8} {'Total':>8} {'Split':>10} {'Status'}")
    print("  " + "-" * 66)

    for cls in all_classes:
        tr = train.get(cls, 0)
        va = val.get(cls, 0)
        total = tr + va
        split_text = f"{tr / total * 100:.0f}/{va / total * 100:.0f}%" if total else "N/A"
        if total == 0:
            status = "EMPTY"
        elif val_target_pct is None:
            status = "OK"
        else:
            current_val_pct = (va / total) * 100
            if abs(current_val_pct - val_target_pct) <= 1.0:
                status = "MATCH"
            else:
                status = "OFF"
        print(f"  {cls:<24} {tr:>8,} {va:>8,} {total:>8,} {split_text:>10} {status}")

    total_train = sum(train.values())
    total_val = sum(val.values())
    grand_total = total_train + total_val

    print("  " + "-" * 66)
    if grand_total > 0:
        print(
            f"  {'TOTAL':<24} {total_train:>8,} {total_val:>8,} {grand_total:>8,} "
            f"{total_train / grand_total * 100:.0f}/{total_val / grand_total * 100:.0f}%"
        )
    else:
        print(f"  {'TOTAL':<24} {0:>8,} {0:>8,} {0:>8,} {'N/A':>10}")

    train_counts = [train.get(c, 0) for c in all_classes if train.get(c, 0) > 0]
    if train_counts and min(train_counts) > 0:
        imbalance_ratio = max(train_counts) / min(train_counts)
        print(f"\n  Train class imbalance (max/min): {imbalance_ratio:.2f}x")

    if grand_total > 0:
        train_pct = (total_train / grand_total) * 100
        val_pct = 100 - train_pct
        print(f"  Overall split: Train {train_pct:.1f}% / Val {val_pct:.1f}%")
        if train_pct < IDEAL_SPLIT_MIN:
            print(f"  [!] Training set is small. Recommended around {IDEAL_SPLIT_MIN}% to {IDEAL_SPLIT_MAX}%.")
        elif train_pct > IDEAL_SPLIT_MAX:
            print("  [!] Validation set is small.")
    print("\n" + "=" * 72 + "\n")


def plot_charts(stats: dict, all_classes: list[str], output_path: Path, title_suffix: str = ""):
    train = stats.get("train", {})
    val = stats.get("val", {})

    tr_counts = [train.get(c, 0) for c in all_classes]
    va_counts = [val.get(c, 0) for c in all_classes]
    tot_counts = [t + v for t, v in zip(tr_counts, va_counts)]

    avg_train = np.mean(tr_counts) if tr_counts else 1
    max_train = max(tr_counts) if tr_counts else 1
    colors = [bar_color(v, avg_train) for v in tr_counts]

    n = len(all_classes)
    y = np.arange(n)
    height = max(5, n * 0.6)

    fig, axes = plt.subplots(1, 3, figsize=(18, height))
    fig.suptitle(f"Dataset Analysis{title_suffix}", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.barh(y, tr_counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axvline(avg_train, color="#E24B4A", linestyle="--", linewidth=1.2)
    ax1.set_title("Train set class distribution", fontsize=11)
    ax1.set_yticks(y)
    ax1.set_yticklabels(all_classes, fontsize=9)
    ax1.set_xlabel("Image count")
    ax1.invert_yaxis()
    patches = [
        mpatches.Patch(color="#378ADD", label="Near average"),
        mpatches.Patch(color="#BA7517", label="Below average"),
        mpatches.Patch(color="#E24B4A", label="Low count"),
        plt.Line2D([0], [0], color="#E24B4A", linestyle="--", label=f"Mean: {avg_train:.0f}"),
    ]
    ax1.legend(handles=patches, fontsize=8, loc="lower right")
    for i, v in enumerate(tr_counts):
        ax1.text(v + max_train * 0.01, i, str(v), va="center", fontsize=7.5)

    ax2 = axes[1]
    ax2.barh(y, tr_counts, label="Train", color="#378ADD", edgecolor="white", linewidth=0.5)
    ax2.barh(y, va_counts, label="Val", color="#1D9E75", edgecolor="white", linewidth=0.5, left=tr_counts)
    ax2.set_title("Train/Val count per class", fontsize=11)
    ax2.set_yticks(y)
    ax2.set_yticklabels(all_classes, fontsize=9)
    ax2.set_xlabel("Image count")
    ax2.invert_yaxis()
    ax2.legend(fontsize=9, loc="lower right")
    for i, (train_count, val_count) in enumerate(zip(tr_counts, va_counts)):
        if train_count > 30:
            ax2.text(
                train_count / 2,
                i,
                str(train_count),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        if val_count > 20:
            ax2.text(
                train_count + val_count / 2,
                i,
                str(val_count),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax3 = axes[2]
    pcts = [t / tot * 100 if tot else 0 for t, tot in zip(tr_counts, tot_counts)]
    pcols = ["#378ADD" if IDEAL_SPLIT_MIN <= p <= IDEAL_SPLIT_MAX else "#BA7517" for p in pcts]
    ax3.barh(y, pcts, color=pcols, edgecolor="white", linewidth=0.5)
    ax3.axvline(IDEAL_SPLIT_MIN, color="#E24B4A", linestyle="--", linewidth=1,
                label=f"Ideal ({IDEAL_SPLIT_MIN}–{IDEAL_SPLIT_MAX}%)")
    ax3.axvline(IDEAL_SPLIT_MAX, color="#E24B4A", linestyle="--", linewidth=1)
    ax3.set_title("Train percentage per class", fontsize=11)
    ax3.set_yticks(y)
    ax3.set_yticklabels(all_classes, fontsize=9)
    ax3.set_xlabel("Train %")
    ax3.set_xlim(0, 110)
    ax3.invert_yaxis()
    ax3.legend(fontsize=8, loc="lower right")
    for i, p in enumerate(pcts):
        ax3.text(p + 1, i, f"{p:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    file_name = "dataset_analysis_after.png" if "After" in title_suffix else "dataset_analysis_before.png"
    out = output_path / file_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Chart saved -> {out}\n")


def ask_val_percentage(default_val_pct: int) -> int:
    user_input = input(f"  Enter validation percentage [default {default_val_pct}] > ").strip()
    if not user_input:
        return default_val_pct
    try:
        value = int(user_input)
    except ValueError:
        print("  Invalid input. Using default value.")
        return default_val_pct

    if value < 5 or value > 50:
        print("  Value out of range (5-50). Using default value.")
        return default_val_pct
    return value


def move_random_files(source_files: list[Path], destination_dir: Path, n_move: int) -> int:
    if n_move <= 0 or not source_files:
        return 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    selected = random.sample(source_files, min(n_move, len(source_files)))
    moved = 0
    for src in selected:
        dest = destination_dir / src.name
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            idx = 1
            while (destination_dir / f"{stem}_mv{idx}{suffix}").exists():
                idx += 1
            dest = destination_dir / f"{stem}_mv{idx}{suffix}"
        shutil.move(str(src), str(dest))
        moved += 1
    return moved


def rebalance_by_val_percentage(base_path: Path, all_classes: list[str], val_percentage: int) -> dict:
    print("\n" + "=" * 72)
    print("  REBALANCING TRAIN/VAL SPLIT")
    print("=" * 72)
    print(f"\n  Target split per class: Train {100 - val_percentage}% / Val {val_percentage}%\n")
    print(f"  {'Class':<24} {'Total':>8} {'Train->Val':>12} {'Val->Train':>12} {'Final Val %':>12}")
    print("  " + "-" * 72)

    for cls in all_classes:
        train_dir = base_path / "train" / cls
        val_dir = base_path / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        train_files = list_images(train_dir)
        val_files = list_images(val_dir)

        total = len(train_files) + len(val_files)
        if total == 0:
            print(f"  {cls:<24} {0:>8,} {0:>12,} {0:>12,} {'N/A':>12}")
            continue

        target_val = int(round(total * val_percentage / 100.0))
        current_val = len(val_files)
        move_to_val = 0
        move_to_train = 0

        if current_val < target_val:
            need = target_val - current_val
            move_to_val = move_random_files(train_files, val_dir, need)
        elif current_val > target_val:
            extra = current_val - target_val
            move_to_train = move_random_files(val_files, train_dir, extra)

        final_train = count_images(train_dir)
        final_val = count_images(val_dir)
        final_total = final_train + final_val
        final_val_pct = (final_val / final_total) * 100 if final_total else 0

        print(
            f"  {cls:<24} {total:>8,} {move_to_val:>12,} {move_to_train:>12,} {final_val_pct:>11.1f}%"
        )

    print("\n  Rebalancing completed.\n")
    return collect_stats(base_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image dataset and rebalance train/val split by random file movement."
    )
    parser.add_argument(
        "--path", type=str, default=".",
        help="Path to dataset root containing train/ and val/"
    )
    parser.add_argument(
        "--val-percentage", type=int, default=None,
        help="Target validation percentage per class (5 to 50)."
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompt and run balancing immediately."
    )
    args = parser.parse_args()

    base_path = Path(args.path).resolve()
    print(f"\nAnalyzing dataset at: {base_path}")

    stats = collect_stats(base_path)
    all_classes = sorted(set(k for s in stats.values() for k in s))
    if not all_classes:
        print("  No classes found. Check your folder structure.")
        return

    # 1) Show existing dataset analysis
    print_report(stats, all_classes)
    plot_charts(stats, all_classes, base_path, title_suffix=" - Before")

    total_train = sum(stats.get("train", {}).values())
    total_val = sum(stats.get("val", {}).values())
    total = total_train + total_val
    current_val_pct = int(round((total_val / total) * 100)) if total else 20
    default_val_pct = max(5, min(50, current_val_pct if current_val_pct > 0 else 20))

    # 2) Ask what validation percentage is needed
    if args.val_percentage is None:
        val_percentage = ask_val_percentage(default_val_pct)
    else:
        if args.val_percentage < 5 or args.val_percentage > 50:
            print("  --val-percentage must be between 5 and 50.")
            return
        val_percentage = args.val_percentage

    if not args.yes:
        answer = input(f"  Proceed to rebalance with Val {val_percentage}%? [y/N] > ").strip().lower()
        if answer not in ("y", "yes"):
            print("\n  Rebalancing cancelled.\n")
            return

    # 3) Move data randomly for each class
    new_stats = rebalance_by_val_percentage(base_path, all_classes, val_percentage)

    # 4) Replot and show analysis again
    print_report(new_stats, all_classes, val_target_pct=val_percentage)
    plot_charts(new_stats, all_classes, base_path, title_suffix=" - After Rebalancing")


if __name__ == "__main__":
    main()