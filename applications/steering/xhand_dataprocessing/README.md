# XHand data processing

Tactile viewer for XHand teleop episodes: three cameras synced with the ten fingertip pressure maps and a force timeline
(force magnitude per fingertip, labelled y axis, peak and mean table). Each fingertip's resting value (median of the first 15 frames) is removed.

```
python build_tactile_viewer.py --episode example/bag_light --episode example/bag_heavy --episode example/episode \
       --labels example/labels.json --out tactile_viewer_example.html
```

Open the HTML in any browser (self-contained, works offline). Several `--episode` folders give a dropdown. `--labels` is optional: it maps a folder name to
`{label, class, split, ...}` shown above the player.

Episode folder:

| File | Content |
|---|---|
| `*.h5` | `{left,right}_hand_tactile_raw` (T, 5, 120, 3), `{left,right}_hand_tactile_force` (T, 5, 3) |
| | optional `{left,right}_hand_target_joint_positions`, `{left,right}_arm_joint_positions`: mark the first and last frame with motion and report idle frames |
| `head_left_rgb.mp4`, `left_wrist.mp4`, `right_wrist.mp4` | cameras (an optional name prefix is fine) |

`make_example_episode.py RAW_EPISODE_DIR OUT_DIR` makes a small episode folder from a raw recording.

Requires numpy, h5py, matplotlib, imageio-ffmpeg.

`../xhand_data/<task>/<episode>/viewer.json` and `viewer.mp4` are the web-format files the project-page viewer loads.
