#!/usr/bin/env python3
"""Move the Vega mobile base forward ~1 meter, as safely as possible.

Built on dexcontrol's Chassis API. The API is *velocity-based*, so we
move a known low speed for a computed duration (distance = speed * time)
and then verify the travelled distance with the wheel encoders.

    pip install dexcontrol

Run:
    python move_one_meter.py                 # forward 1.0 m at 0.1 m/s
    python move_one_meter.py --distance 0.5  # forward 0.5 m
    python move_one_meter.py --distance -1.0 # backward 1.0 m

SAFETY: keep a hand on the physical e-stop. Start with the smallest
distance/speed and a clear floor before trusting any number here.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from loguru import logger

from dexcontrol.robot import Robot

# Conservative defaults. Slow is safe: at 0.1 m/s you have ~10 s of
# reaction time over a meter, and timing error scales with speed.
DEFAULT_SPEED = 0.1          # m/s
MAX_SPEED = 0.25             # m/s — hard cap, refuse anything faster
WHEEL_RADIUS = 0.0           # set if you want encoder distance check (m); 0 disables


def estimate_distance_from_encoders(start: np.ndarray, end: np.ndarray) -> float | None:
    """Rough straight-line distance from wheel encoder deltas (radians).

    Returns None if WHEEL_RADIUS isn't configured. This is a sanity check,
    not ground truth — slip and steering geometry are ignored.
    """
    if WHEEL_RADIUS <= 0.0:
        return None
    avg_wheel_rotation = float(np.mean(np.abs(end - start)))  # rad
    return avg_wheel_rotation * WHEEL_RADIUS


def move_distance(distance: float, speed: float) -> None:
    """Move the base straight by `distance` meters at `speed` m/s."""
    speed = abs(speed)
    if speed > MAX_SPEED:
        raise ValueError(f"speed {speed} m/s exceeds safety cap {MAX_SPEED} m/s")
    if speed <= 0:
        raise ValueError("speed must be > 0")

    direction = math.copysign(1.0, distance)   # +1 forward, -1 backward
    duration = abs(distance) / speed           # seconds to travel the distance
    signed_speed = direction * speed

    logger.warning("Be ready to press the e-stop. Floor must be clear.")
    confirm = input(
        f"Move base {distance:+.2f} m at {speed:.2f} m/s "
        f"(~{duration:.1f} s)? [y/N] "
    )
    if confirm.strip().lower() != "y":
        logger.info("Aborted by user.")
        return

    with Robot() as bot:
        chassis = bot.chassis

        # Make sure the base is actually publishing state before we command it.
        if not chassis.wait_for_active(timeout=5.0):
            logger.error("Chassis not active (no state data). Aborting.")
            return

        start_enc = np.array(chassis.wheel_encoder_pos)
        try:
            # move_straight blocks for wait_time, then the next line runs.
            chassis.move_straight(signed_speed, wait_time=duration)
        finally:
            # Always stop, even on Ctrl-C / exception, before we exit.
            chassis.stop()
            logger.info("Stop command sent.")

        end_enc = np.array(chassis.wheel_encoder_pos)
        measured = estimate_distance_from_encoders(start_enc, end_enc)
        if measured is not None:
            logger.info(
                f"Commanded {abs(distance):.2f} m, encoders suggest "
                f"~{measured:.2f} m travelled."
            )
        else:
            logger.info(
                "Set WHEEL_RADIUS at top of file to enable encoder distance check."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Move Vega base ~1 m safely.")
    parser.add_argument(
        "--distance", type=float, default=1.0,
        help="meters to move; negative = backward (default: 1.0)",
    )
    parser.add_argument(
        "--speed", type=float, default=DEFAULT_SPEED,
        help=f"speed in m/s, capped at {MAX_SPEED} (default: {DEFAULT_SPEED})",
    )
    args = parser.parse_args()
    move_distance(args.distance, args.speed)


if __name__ == "__main__":
    main()
