"""
Synchronization Strategies — SP / SSP / BAP (Paper Section 3.3)
==================================================================
"Synchronization defines how often global model updates are aggregated
within the telemedicine BC-FL system. Balancing convergence speed with
communication efficiency is essential for scalable deployment."

Strategies implemented
-----------------------
SP  (Synchronous Parallel)
    All trainers submit updates within a fixed round deadline. A strict
    deadline marks the end of each round; submissions arriving after the
    deadline are DROPPED for this round (ensuring consistency among
    updates but requiring slower nodes to wait / be excluded).

SSP (Stale Synchronous Parallel)
    Introduces a controlled "slack ratio" allowing some trainers to
    continue beyond the deadline. Submissions within
    [deadline, deadline * (1 + slack_ratio)] are accepted but
    DOWN-WEIGHTED in FedAvg proportionally to how late they are
    ("staleness penalty"). This balances convergence accuracy with
    reduced waiting time.

BAP (Barrierless Asynchronous Parallel)
    Trainers send updates as they finish, with NO deadline/barrier,
    minimizing idle time and improving bandwidth utilization. A mild
    staleness penalty (relative to the slowest node in the round) is
    still applied so a single very slow/delayed update cannot dominate
    or degrade the global model.

The Administrator (fl/roles.py) can dynamically switch between these
modes based on network performance and node-participation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple


class SyncManager:
    """
    Decides, for a single FL round, which miner submissions are INCLUDED
    in aggregation and what STALENESS WEIGHT (in [0, 1]) each included
    submission receives.

    Parameters
    ----------
    mode              : "SP" | "SSP" | "BAP"
    base_deadline_sec : Administrator-configured round deadline (Phase A)
    slack_ratio       : SSP slack ratio (fraction of deadline allowed extra)
    """

    def __init__(self, mode: str = "SSP", base_deadline_sec: float = 0.05,
                 slack_ratio: float = 0.5):
        mode = mode.upper()
        if mode not in ("SP", "SSP", "BAP"):
            raise ValueError(f"Unknown sync mode: {mode!r}")
        self.mode              = mode
        self.base_deadline_sec = base_deadline_sec
        self.slack_ratio       = slack_ratio

    def process_submissions(
        self, submissions: Dict[int, dict]
    ) -> Tuple[List[int], Dict[int, float], Dict[int, int]]:
        """
        submissions: {trainer_id: {"latency_sec": float, ...}}

        Returns
        -------
        included_ids : trainer ids whose updates participate in FedAvg
        staleness_w  : {trainer_id: weight multiplier in [0, 1]}
        late_flags   : {trainer_id: 0 (on-time) | 1 (late/stale)}
        """
        latencies = {tid: s.get("latency_sec", 0.0) for tid, s in submissions.items()}
        max_lat   = max(latencies.values()) if latencies else 0.0

        included:  List[int]       = []
        stale_w:   Dict[int, float] = {}
        late_flag: Dict[int, int]   = {}

        if self.mode == "SP":
            deadline = self.base_deadline_sec
            for tid, lat in latencies.items():
                if deadline <= 0 or lat <= deadline:
                    included.append(tid)
                    stale_w[tid]   = 1.0
                    late_flag[tid] = 0
                else:
                    # Synchronous barrier: late updates dropped this round
                    stale_w[tid]   = 0.0
                    late_flag[tid] = 1

        elif self.mode == "SSP":
            deadline = self.base_deadline_sec
            slack    = deadline * (1.0 + self.slack_ratio)
            for tid, lat in latencies.items():
                if deadline <= 0 or lat <= deadline:
                    included.append(tid)
                    stale_w[tid]   = 1.0
                    late_flag[tid] = 0
                elif lat <= slack:
                    included.append(tid)
                    frac          = (lat - deadline) / max(slack - deadline, 1e-9)
                    stale_w[tid]  = max(0.3, 1.0 - 0.7 * frac)
                    late_flag[tid] = 1
                else:
                    stale_w[tid]   = 0.0
                    late_flag[tid] = 1

        else:  # BAP — no barrier, mild relative staleness penalty
            for tid, lat in latencies.items():
                included.append(tid)
                rel            = (lat / max_lat) if max_lat > 0 else 0.0
                stale_w[tid]   = max(0.5, 1.0 - 0.5 * rel)
                late_flag[tid] = 1 if rel > 0.6 else 0

        return included, stale_w, late_flag

    def info(self) -> dict:
        return {
            "mode":              self.mode,
            "base_deadline_sec": self.base_deadline_sec,
            "slack_ratio":       self.slack_ratio,
        }

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode not in ("SP", "SSP", "BAP"):
            raise ValueError(f"Unknown sync mode: {mode!r}")
        self.mode = mode
