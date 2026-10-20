"""Synthetic study of the paper (Section 4.1, Figures fig_synth1 and fig_synth2): robots exchanging places in
pairs among static spheres (Problem B), GlobMP against RRT-Connect, RRT*, GCS and SDP-MP, and a scaling study
from 2 to 200 robots.

    instances  instance generator, program, route classes and their waypoint openers, exact evaluation
    run        GlobMP on every route class of a main-comparison instance
    scale      GlobMP end to end on a scaling instance (time to feasibility, time per iteration)
    baselines  RRT-Connect, RRT*, GCS on IRIS regions, SDP-MP, and the post-processing of planner paths
    report     tables and quoted numbers from the result jsons
    figures    fig_synth1 and fig_synth2

See README.md in this directory for the commands.
"""
