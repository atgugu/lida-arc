#!/usr/bin/env python
"""
Download real ARC-AGI dataset tasks from GitHub.

Downloads 100 tasks from the official ARC-AGI repository:
https://github.com/fchollet/ARC-AGI
"""

import json
import urllib.request
import os
from pathlib import Path

def download_arc_tasks(output_dir, num_training=50, num_eval=50):
    """Download ARC tasks from GitHub."""

    output_dir = Path(output_dir)
    training_dir = output_dir / "arc_100" / "training"
    eval_dir = output_dir / "arc_100" / "evaluation"

    training_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # GitHub raw URLs for ARC-AGI dataset
    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

    print("Downloading ARC-AGI dataset...")
    print(f"Target: {num_training} training + {num_eval} evaluation tasks")
    print()

    # First, get list of available tasks by trying to download challenges/tasks.json
    # If that doesn't work, we'll use a known list

    # Known task IDs from ARC-AGI (sample of 100)
    training_task_ids = [
        "007bbfb7", "00d62c1b", "017c7c7b", "025d127b", "0520fde7",
        "05269061", "05f2a901", "06df4c85", "08ed6ac7", "09629e4f",
        "0962bcdd", "0a938d79", "0b148d64", "0ca9ddb6", "0d3d703e",
        "0dfd9992", "0e206a2e", "1190e5a7", "11852cab", "1277c52d",
        "128cd6b0", "1367f84f", "137eaa0f", "150deff5", "1571fe93",
        "15663ba9", "178fcbfb", "17cae0c1", "19bb5feb", "1a07d186",
        "1b2d62fb", "1b60fb0c", "1bfc4729", "1c79d348", "1c90dcb5",
        "1caeab9d", "1cf80156", "1e0a9b12", "1e32b0e9", "1e9febe2",
        "1f0c79e5", "1f642eb9", "1f85a75f", "1f876c06", "1fad071e",
        "2013d3e2", "20818e16", "21f83797", "22168020", "22233c11"
    ]

    eval_task_ids = [
        "23581191", "23b5c85d", "253bf280", "25d487eb", "25d8a9c8",
        "25ff71a9", "264363fd", "272f95fa", "27a77e38", "28bf18c6",
        "28e73c20", "29623f79", "29c11459", "29ec7d0e", "2bcee788",
        "2bee17df", "2c608aff", "2dd70a9a", "2dee498d", "2f876c35",
        "31aa019c", "321b1fc6", "32597951", "3428a4f5", "3618c87e",
        "3631a71a", "363442ee", "36d67576", "36fdfd69", "39a8645d",
        "39e1d7f9", "3aa6fb7a", "3ac3eb23", "3ad63a2d", "3af2c5a8",
        "3bd67248", "3bdb4ada", "3c9b0459", "3de23699", "3e980e27",
        "3f7978a0", "40853293", "41e4d17e", "4258a5f9", "4290ef0e",
        "44d8ac46", "44f52bb0", "4522001f", "4612dd53", "46f33fce"
    ]

    downloaded_train = 0
    downloaded_eval = 0

    # Download training tasks
    print("Downloading training tasks...")
    for i, task_id in enumerate(training_task_ids[:num_training]):
        try:
            url = f"{base_url}/training/{task_id}.json"
            filepath = training_dir / f"{task_id}.json"

            urllib.request.urlretrieve(url, filepath)
            downloaded_train += 1

            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i + 1}/{num_training} training tasks")

        except Exception as e:
            print(f"  Failed to download {task_id}: {e}")

    print(f"✓ Downloaded {downloaded_train} training tasks\n")

    # Download evaluation tasks
    print("Downloading evaluation tasks...")
    for i, task_id in enumerate(eval_task_ids[:num_eval]):
        try:
            url = f"{base_url}/evaluation/{task_id}.json"
            filepath = eval_dir / f"{task_id}.json"

            urllib.request.urlretrieve(url, filepath)
            downloaded_eval += 1

            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i + 1}/{num_eval} evaluation tasks")

        except Exception as e:
            print(f"  Failed to download {task_id}: {e}")

    print(f"✓ Downloaded {downloaded_eval} evaluation tasks\n")

    print("="*60)
    print(f"Total downloaded: {downloaded_train + downloaded_eval} tasks")
    print(f"  Training: {downloaded_train}")
    print(f"  Evaluation: {downloaded_eval}")
    print("="*60)

    return downloaded_train, downloaded_eval


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    download_arc_tasks(data_dir, num_training=50, num_eval=50)
