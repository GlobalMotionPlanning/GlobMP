"""
Single source of truth for every (map type, robot count) configuration in
this extended reproduction. Each entry fully specifies what environment/
robot/map_name a scenario uses so make_instances / generate_data / train_smd
/ evaluate_smd can all be driven generically instead of copy-pasted per case.
"""

ROBOT_CLASS_BY_N = {
    2: 'RobotCompositeTwoPlanarDisk',
    3: 'RobotCompositeThreePlanarDisk',
    6: 'RobotCompositeSixPlanarDisk',
    9: 'RobotCompositeNinePlanarDisk',
    12: 'RobotCompositeTwelvePlanarDisk',
    15: 'RobotCompositeFifteenPlanarDisk',
    18: 'RobotCompositeEighteenPlanarDisk',
    21: 'RobotCompositeTwentyOnePlanarDisk',
}

# env_id is the *base* name; the dataset/env loader appends 'ExtraObjects' itself.
SCENARIOS = [
    # ---- Random maps: well-specified by the paper (0/10/20 random circular obstacles) ----
    dict(scenario='empty', num_agents=3, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),
    dict(scenario='empty', num_agents=6, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),
    dict(scenario='empty', num_agents=9, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),

    dict(scenario='empty', num_agents=12, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),
    dict(scenario='empty', num_agents=15, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),
    dict(scenario='empty', num_agents=18, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),
    dict(scenario='empty', num_agents=21, env_id='EnvEmptyNoWait2D', map_name='empty_map', num_obstacles=0),
    dict(scenario='basic', num_agents=3, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),
    dict(scenario='basic', num_agents=6, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),
    dict(scenario='basic', num_agents=9, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),

    dict(scenario='basic', num_agents=12, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),
    dict(scenario='basic', num_agents=15, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),
    dict(scenario='basic', num_agents=18, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),
    dict(scenario='basic', num_agents=21, env_id='EnvEmptyNoWait2D', map_name='basic_map', num_obstacles=10),
    dict(scenario='dense', num_agents=3, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),
    dict(scenario='dense', num_agents=6, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),
    dict(scenario='dense', num_agents=9, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),

    dict(scenario='dense', num_agents=12, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),
    dict(scenario='dense', num_agents=15, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),
    dict(scenario='dense', num_agents=18, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),
    dict(scenario='dense', num_agents=21, env_id='EnvEmptyNoWait2D', map_name='dense_map', num_obstacles=20),
    # ---- Practical maps: fixed wall/box geometry already baked into these repo
    # environments (built by the same group for the related MMD paper). Used here
    # as stand-ins for the paper's Corridor/Room/Shelf -- not the paper's own
    # released benchmark geometry, which isn't in this codebase.
    dict(scenario='corridor', num_agents=2, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    # 3/6/9-robot Conveyor cells: needed to complete the paper's multi-robot
    # table (Empty/Highways/Conveyor/Drop-Region x {3,6,9}), same env as
    # corridor-2 above, just more agents -- corridor-2 was SMD's own
    # 2-robot "Corridor Maps" benchmark cell and doesn't cover this.
    dict(scenario='corridor', num_agents=3, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='corridor', num_agents=6, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='corridor', num_agents=9, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='corridor', num_agents=12, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='corridor', num_agents=15, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='corridor', num_agents=18, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='corridor', num_agents=21, env_id='EnvConveyor2D', map_name='corridor_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=3, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=6, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=9, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=12, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=15, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=18, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='room', num_agents=21, env_id='EnvHighways2D', map_name='room_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=3, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=6, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=9, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=12, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=15, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=18, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
    dict(scenario='shelf', num_agents=21, env_id='EnvDropRegion2D', map_name='shelf_map', num_obstacles=0,
         threshold_start_goal_pos=0.6),
]


# ---------------------------------------------------------------------------------
# SCENARIO-1 ADDITIONS
#
# Two maps rebuilt from SMD's own Figure 3, which ships only one of them as code:
#   corridorbay  -- Fig. 3a. Three parallel single-file channels (0.165 wide: one robot
#                   at 2r=0.10 fits, two at 0.20 do not) joined by an open cross-roads,
#                   with staging areas top and bottom. The shipped env_narrow_corridors
#                   is NOT single-file (channel 0.328) and a single narrow channel cannot
#                   hold 21 robots, hence three of them.
#   crossroom    -- Fig. 3c. A plus-shaped divider whose arms stop short of the boundary,
#                   so four quadrants connect only through four edge gaps.
#
# HARD VARIANTS. Every cell also gets a '<scenario>hard' twin carrying obs_scale, chosen
# per map as the largest inflation where two robots still pass abreast WITH THE WORKSPACE
# WALL COUNTED (see tune_hard_settings.py). Three maps cannot be inflated at all -- Basic
# and Dense already have wall gaps under two robots at the shipped setting, and the
# corridor is single-file by design -- so their hard twin keeps obs_scale 1.0 and their
# difficulty comes from team size alone.
#
# The twin needs its own scenario NAME because dataset_subdir and model_id are derived
# from it; without that, training a hard-row model would overwrite the shipped one.
_NEW_MAPS = [
    ('corridorbay', 'EnvCorridorSMD2D', 'corridorbay_map', 0),
    ('crossroom',   'EnvCrossRoom2D',   'crossroom_map',   0),
]
_TEAM_SIZES = [3, 6, 9, 12, 15, 18, 21]
for _sc, _env, _mn, _nobs in _NEW_MAPS:
    for _n in _TEAM_SIZES:
        SCENARIOS.append(dict(scenario=_sc, num_agents=_n, env_id=_env,
                              map_name=_mn, num_obstacles=_nobs,
                              threshold_start_goal_pos=0.6))

# obstacle scale for the HARD row, per base map; 1.0 means "cannot be inflated"
# Set EMPIRICALLY, then CORRECTED for reachability. corridor and crossroom were 2.20,
# at which the free space is no longer connected at the planner's clearance (0.105):
# corridor splits into 2 regions, crossroom into 4 sealed quadrants. Measured consequence
# on crossroom x2.20: PP generation yielded 5 demos from 775 attempts, and raising the
# single-agent dataset 133x (205 -> 27,311 demos) moved that only to 1.0% -- because no
# path exists to learn. Connectivity ceilings by bisection: corridor 1.57, crossroom 1.12;
# both stepped back one notch below. The gate now lives in run_scenario1_pilot.sh.
#
# The earlier values came from an r_crit criterion that was wrong twice over -- it used
# the nominal robot radius 0.05 when the planner enforces 0.105, and it measured global
# free-space connectivity, which reports "impassable" for shipped maps that demonstrably
# work with 21 robots. The validator is the thing that actually gates instance generation,
# so it sets the values.
HARD_OBS_SCALE = {
    'empty': 1.0,           # no obstacles to scale; its axis is the robot radius below
    'basic': 2.20,
    # dense stays at its PUBLISHED 1.0. At 1.8 only 4/20 instances at N=3 and 0/20 above
    # have a solution at all -- the inflation disconnects the map for multi-robot
    # planning. At 1.0 it is 20/20 at every team size. (An earlier reading said dense was
    # broken even at 1.0; that was an artifact of certifying against the planner's
    # obstacle_cutoff_margin rather than the radius the scorer uses.)
    'dense': 1.00,
    'corridor': 1.50, 'room': 1.45, 'shelf': 1.60,
    'corridorbay': 1.15, 'crossroom': 1.10,
}
# the Empty map has no obstacles, so its axis is the robot radius instead
HARD_ROBOT_R = {'empty': 0.10}

# ---- Robotarium port -------------------------------------------------------------
# The same layouts read as METRES on the Georgia Tech testbed: its arena's y-range is
# exactly [-1, 1], so a layout defined on [-1, 1]^2 drops in at scale 1.0 with 0.6 m of
# spare x on either side. What changes is the hardware, not the geometry:
#   robot 0.055 m radius (0.11 m diameter GRITSBot) against our 0.05
#   separation 0.25 m planned, 0.185 m hardware threshold, against our 2r = 0.11
#   duration 30 s -- a 1.6 m swap at the testbed's 0.08 m/s planned peak takes 20 s
#   team sizes to 18, the testbed's maximum
# Separate scenario names because dataset_subdir and model_id derive from them: a model
# trained for a 0.055 m robot must not be reachable under the 0.05 m name.
ROBOTARIUM_ROBOT_R = 0.055
ROBOTARIUM_MAX_N = 18
# maps whose obstacle scale had to come down for an 18-robot team to fit at 0.25 m
# separation; measured, not chosen (see the probe in A.2.2). The rest keep their scale.
# MEASURED against the testbed's own problem, not chosen: the 0.80 m antipodal ring
# with an 0.055 m robot admits instances at every team size 3-18 only at these
# scales (probe: 60 rotations per (map, N), endpoints clear by 2.1*r and layer-1
# feasible). room is ABSENT because no scale works -- OBS_SCALE shrinks each box
# about its own centre, so Highways' tiled lane walls become dotted lines that
# still cross the ring at the same angles; ring clearance saturates at 81% as the
# scale goes to zero and N=9/15/18 never place.
ROBOTARIUM_OBS_SCALE = {'corridor': 1.1, 'crossroom': 0.85, 'shelf': 0.4,
                        'corridorbay': 0.2, 'empty': 1.0}

_rbt = []
for _cfg in list(SCENARIOS):
    if _cfg['num_agents'] > ROBOTARIUM_MAX_N:
        continue
    _r = dict(_cfg)
    _r['scenario'] = _cfg['scenario'] + 'rbt'
    _r['base_scenario'] = _cfg['scenario']
    _r['obs_scale'] = ROBOTARIUM_OBS_SCALE.get(
        _cfg['scenario'], HARD_OBS_SCALE.get(_cfg['scenario'], 1.0))
    _r['robot_r'] = ROBOTARIUM_ROBOT_R
    _rbt.append(_r)

_base = list(SCENARIOS)
for _cfg in _base:
    _h = dict(_cfg)
    _h['scenario'] = _cfg['scenario'] + 'hard'
    _h['base_scenario'] = _cfg['scenario']
    _h['obs_scale'] = HARD_OBS_SCALE.get(_cfg['scenario'], 1.0)
    if _cfg['scenario'] in HARD_ROBOT_R:
        _h['robot_r'] = HARD_ROBOT_R[_cfg['scenario']]
    SCENARIOS.append(_h)

SCENARIOS.extend(_rbt)

for _cfg in SCENARIOS:
    _cfg.setdefault('obs_scale', 1.0)
    _cfg.setdefault('base_scenario', _cfg['scenario'])

for cfg in SCENARIOS:
    cfg.setdefault('threshold_start_goal_pos', 1.0)
    cfg['robot_id'] = ROBOT_CLASS_BY_N[cfg['num_agents']]
    # NOTE: env_id alone collides across empty/basic/dense (all use EnvEmptyNoWait2D,
    # differing only in map_name's obstacle density) -- must prefix with scenario
    # name or their data_trajectories/data_trained_models dirs would overwrite
    # each other. empty-3 is kept unprefixed since it matches the already-completed
    # first run's directory name.
    if cfg['scenario'] == 'empty' and cfg['num_agents'] == 3:
        cfg['dataset_subdir'] = f"{cfg['env_id']}-{cfg['robot_id']}"
    else:
        cfg['dataset_subdir'] = f"{cfg['scenario']}_{cfg['env_id']}-{cfg['robot_id']}"
    cfg['model_id'] = cfg['dataset_subdir']
    cfg['tag'] = f"{cfg['scenario']}-{cfg['num_agents']}"


def get_scenario(tag):
    for cfg in SCENARIOS:
        if cfg['tag'] == tag:
            return cfg
    raise KeyError(f"Unknown scenario tag: {tag}")


if __name__ == '__main__':
    for cfg in SCENARIOS:
        print(cfg['tag'], '->', cfg['dataset_subdir'])
