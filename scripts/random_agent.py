"""Run a random agent on LunarLander-v3 and save a sample video.

Runs a handful of episodes with uniformly random actions, prints the labelled
first-frame observation of each episode together with the total reward and
length, and records the first episode to ``videos/random_agent.mp4``.
"""

from __future__ import annotations

import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

ENV_ID = "LunarLander-v3"
N_EPISODES = 5
VIDEO_DIR = "videos"
VIDEO_NAME = "random_agent.mp4"

OBS_LABELS: tuple[str, ...] = (
    "x position",
    "y position",
    "x velocity",
    "y velocity",
    "angle",
    "angular velocity",
    "left leg contact",
    "right leg contact",
)


def main() -> None:
    """Run random episodes and save the first episode as a video."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(
        env,
        VIDEO_DIR,
        name_prefix="random_agent",
        episode_trigger=lambda episode: episode == 0,
        disable_logger=True,
    )

    for episode in range(N_EPISODES):
        obs, _ = env.reset(seed=episode)
        print(f"\nEpisode {episode}")
        for label, value in zip(OBS_LABELS, obs):
            print(f"  {label:<18} = {value:+.4f}")

        total_reward = 0.0
        length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total_reward += reward
            length += 1
        print(f"  -> total reward: {total_reward:+.2f}, length: {length}")

    env.close()

    # RecordVideo writes "random_agent-episode-0.mp4"; rename to deliverable.
    produced = os.path.join(VIDEO_DIR, "random_agent-episode-0.mp4")
    target = os.path.join(VIDEO_DIR, VIDEO_NAME)
    if os.path.exists(produced):
        os.replace(produced, target)
    print(f"\nSaved video: {target}")


if __name__ == "__main__":
    main()
