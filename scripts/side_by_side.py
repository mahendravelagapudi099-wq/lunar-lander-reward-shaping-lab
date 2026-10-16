"""Stack two recorded landing videos side by side (hover bonus vs potential).

Reads ``videos/<mode_a>_seed<S>.mp4`` and ``videos/<mode_b>_seed<S>.mp4`` (as
written by ``scripts/record_video.py --mode``) and writes one clip with the
two episodes next to each other -- by default hover_bonus on the left and
potential_shaped on the right -- so the farming behaviour (or its absence) is
visible at a glance. The two clips keep their own durations: the composite
runs as long as the longer episode and pads the shorter side with black.

Example::

    .venv\\Scripts\\python.exe scripts\\side_by_side.py --seed 0
    .venv\\Scripts\\python.exe scripts\\side_by_side.py --mode-a hover_bonus --mode-b potential_shaped --seed 1
"""

from __future__ import annotations

import argparse
import os

from moviepy import VideoFileClip, clips_array

VIDEO_DIR = "videos"
DEFAULT_A = "hover_bonus"
DEFAULT_B = "potential_shaped"


def stack(left_path: str, right_path: str, out_path: str) -> str:
    """Write the two clips side by side to ``out_path`` and return it."""
    left = VideoFileClip(left_path)
    right = VideoFileClip(right_path)
    if left.h != right.h:
        # Recordings of the same env always match in size; guard anyway.
        right = right.resized(height=left.h)
    final = clips_array([[left, right]])
    final.write_videofile(out_path, codec="libx264", logger=None)
    left.close()
    right.close()
    final.close()
    print(f"Saved: {out_path} ({left.w + right.w}x{left.h}, {final.duration:.1f}s)")
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="which seed's videos to stack (default: 0)")
    parser.add_argument("--mode-a", default=DEFAULT_A, help="left panel mode (default: hover_bonus)")
    parser.add_argument("--mode-b", default=DEFAULT_B, help="right panel mode (default: potential_shaped)")
    parser.add_argument("--video-dir", default=VIDEO_DIR, help="directory holding the recorded videos")
    parser.add_argument("--out", default=None, help="output path (default: <video-dir>/side_by_side_<a>_vs_<b>_seed<S>.mp4)")
    return parser.parse_args()


def main() -> None:
    """Find the two episode videos and stack them."""
    args = parse_args()
    left = os.path.join(args.video_dir, f"{args.mode_a}_seed{args.seed}.mp4")
    right = os.path.join(args.video_dir, f"{args.mode_b}_seed{args.seed}.mp4")
    for path in (left, right):
        if not os.path.exists(path):
            raise SystemExit(
                f"Missing episode video: {path}\n"
                f"Record it first, e.g. scripts/record_video.py --mode "
                f"{os.path.basename(path).rsplit('_seed', 1)[0]} --seeds {args.seed}"
            )
    out = args.out or os.path.join(
        args.video_dir, f"side_by_side_{args.mode_a}_vs_{args.mode_b}_seed{args.seed}.mp4"
    )
    stack(left, right, out)


if __name__ == "__main__":
    main()