# Steering tactile-reactive dexterous policies (Section 4.3)

GlobMP steers the action chunks of a learned policy (GR00T N1.7 and pi_0.5, with and without tactile
input) on a Dexmate Vega-1 with two XHAND1 hands. A meta-reasoner sets the active constraint blocks once
per episode (keep-out, keep-in, grip); a certifier solves (P) per chunk with the policy's chunk as the
reference curve (`Problem.set_tracking` / `set_tracking_fine`), so the executed curve meets every
active block at every t (hand-position blocks up to the linearization of the forward kinematics).

The robot-side code (policy servers, tactile pipeline, forward kinematics, demonstrated limits and the
run logs behind Figure 6) runs on the robot workstation and is not yet included in this repository.
