from dataclasses import dataclass
import math
from typing import List, Tuple, Optional


@dataclass
class EpisodeMetrics:
    reached_goal: bool
    steps: int
    time_sec: float
    collisions: int
    min_goal_dist: float
    mean_speed: float


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_metrics(
    positions: List[Tuple[float, float]],
    speeds: List[float],
    collisions: int,
    goal: Tuple[float, float],
    dt: float,
    goal_radius: float = 20.0,
) -> EpisodeMetrics:
    if not positions:
        return EpisodeMetrics(False, 0, 0.0, collisions, float("inf"), 0.0)

    dists = [dist(p, goal) for p in positions]
    min_goal_dist = min(dists)
    reached = min_goal_dist <= goal_radius

    steps = len(positions)
    time_sec = steps * dt
    mean_speed = sum(speeds) / max(1, len(speeds))

    return EpisodeMetrics(
        reached_goal=reached,
        steps=steps,
        time_sec=time_sec,
        collisions=collisions,
        min_goal_dist=min_goal_dist,
        mean_speed=mean_speed,
    )
