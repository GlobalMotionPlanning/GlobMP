# XHand teleop data

Four tasks recorded on a bimanual robot with two XHand1 hands (12 DOF each, fingertip tactile), glove teleoperation, four RGB cameras (640×360, 30 fps).
Each episode folder holds one `.h5` (hand and arm joints, poses, tactile force and pressure maps), four camera videos and `result.json`.

| Folder | Task | Episodes | Train | Held out |
|---|---|---|---|---|
| `pick_cup_full_empty_random_1014` | right hand picks a full or an empty paper cup, positions randomized | 100 | 100 | 0 |
| `handover_random_1007` | pick a tissue box on a shelf, hand it to the other hand, place it on another level | 104 | 100 | 4 |
| `tissue_random_1009` | hold a tissue box with the left hand, pull a tissue with the right | 102 | 91 | 11 |
| `bag_box_1019` | lift a handbag with both hands and place it in a box; light and heavy bags | 128 | 100 | 28 |

`<task>/splits.json` lists the train and held-out episode names; for `pick_cup_full_empty_random_1014` and `bag_box_1019` it also gives the class of each episode
(full / empty cup, light / heavy bag). `<task>/DATASET_CARD.txt` has the details of each recording.

Notes
- `pick_cup_full_empty_random_1014`: only the usable episodes are included (full cup 0060–0109, empty cup 0110–0168); the load class comes from the right-hand fingertip force.
- `bag_box_1019`: most episodes are robot replays of one earlier recording with scripted hand commands and randomized bag and box poses, so the hand does not react to touch. Light and heavy are labelled by grip force (less force = light).
- In the cup sessions the file `left_wrist.mp4` shows the right hand and `right_wrist.mp4` the left.
- Tactile layout: force `(T, 5, 3)` per hand (thumb to pinky, fx fy fz); pressure maps `(T, 5, 120, 3)`, a 10 × 12 grid per fingertip. Fingertip sensors are not zeroed before recording; subtract each episode's resting value.

Each episode folder also has `viewer.json` and `viewer.mp4`, the small web-format files (maps and force, and the three cameras in one video) that the project-page viewer loads; the raw files are on Hugging Face.

Viewer code: `../xhand_dataprocessing/`.
