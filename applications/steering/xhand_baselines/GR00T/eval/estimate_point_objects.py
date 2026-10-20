#!/usr/bin/env python3
"""Estimate 3D table positions of the point-task objects from the head camera.

Pipeline: HSV blob detection on the episode's first frame -> ray through each blob
centroid using the zed_left_camera pose (FK via Genesis from the GT frame-0 joints)
-> intersect the table plane (z = TABLE_TOP_Z) -> {name: [x, y]} JSON.

Positions are ESTIMATES (approximate ZED intrinsics, plane assumption) — good enough
to place illustrative objects in replay videos, not for metric evaluation. An
annotated overlay png is written next to the JSON for visual verification.
"""
import argparse
import json
import os

import cv2
import numpy as np

# object color ranges in HSV (OpenCV H in [0,180]) — tuned on the 260708 scene
COLORS = {
    "red_block":   [(np.array([0, 120, 90]), np.array([8, 255, 255])),
                    (np.array([170, 120, 90]), np.array([180, 255, 255]))],
    "green_block": [(np.array([40, 50, 40]), np.array([90, 255, 220]))],
    # navy block on a blue-reflecting metal table: only slightly more saturated and
    # darker than the table itself (block S~95 V~65-110 vs table S~78 V~123).
    # lighting varies between sessions — the medium-blue fallback window is tried
    # only when the dark window finds nothing (see detect_blobs).
    "blue_block":  [(np.array([95, 85, 40]), np.array([120, 180, 115]))],
    "blue_block_alt": [(np.array([95, 120, 116]), np.array([125, 255, 190]))],
    "yellow_block": [(np.array([20, 120, 120]), np.array([35, 255, 255]))],
}
DRAW = {"red_block": (0, 0, 255), "green_block": (0, 200, 0),
        "blue_block": (255, 80, 0), "yellow_block": (0, 220, 220),
        "red_marker": (180, 105, 255)}

TABLE_TOP_Z = float(os.environ.get("TABLE_TOP_Z", 0.72))
# approx ZED intrinsics at 960x600 (HD1200 halved)
FX = FY = float(os.environ.get("CAM_FX", 534.0))
CX, CY = 480.0, 300.0


def _rect_angle(c):
    """image angle (deg) of the contour's major axis via minAreaRect."""
    (cx, cy), (rw, rh), a = cv2.minAreaRect(c)
    return a if rw >= rh else a + 90.0


def detect_blobs(bgr, table_mask):
    """-> {name: (u, v, angle_img_deg)} pixel bottom-centers + major-axis angle."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = {}
    red_boxes = []
    for name, ranges in COLORS.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        m &= table_mask
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = 15 if name == "red_block" else 40
        W_img = hsv.shape[1]
        cnts = [c for c in cnts if cv2.contourArea(c) > min_a]
        if name == "blue_block_alt" and "blue_block" in out:
            continue                                   # dark window already found it
        if name.startswith("blue_block"):
            # navy block is a compact central blob; metal table reflects the blue
            # curtain in broad streaks near the edges — require shape + central band
            def ok(c):
                x, y, w, h = cv2.boundingRect(c)
                return (0.36 * W_img < x + w / 2 < 0.70 * W_img
                        and 0.5 < w / max(1, h) < 3.2 and 10 < w < 70)
            cnts = [c for c in cnts if ok(c)]
        if not cnts:
            continue
        cnts.sort(key=cv2.contourArea, reverse=True)
        if name == "red_block":
            red_boxes = [cv2.boundingRect(c) for c in cnts[:2]]
            red_cnts = cnts[:2]
        else:
            x, y, w, h = cv2.boundingRect(cnts[0])
            out[name.replace("_alt", "")] = (x + w / 2, y + h, _rect_angle(cnts[0]))
    # two red blobs: the block is the larger, the marker's red cap the smaller
    if red_boxes:
        red_pairs = sorted(zip(red_boxes, red_cnts), key=lambda p: p[0][2] * p[0][3], reverse=True)
        (x, y, w, h), c = red_pairs[0]
        out["red_block"] = (x + w / 2, y + h, _rect_angle(c))
        if len(red_pairs) > 1:
            (x, y, w, h), c = red_pairs[1]
            out["red_marker"] = (x + w / 2, y + h, _rect_angle(c), w, h)
    if "red_marker" not in out:
        # marker fallback 1: its pink-red cap (small, low-saturation red)
        pink = ((cv2.inRange(hsv, np.array([160, 40, 120]), np.array([180, 160, 255]))
                 | cv2.inRange(hsv, np.array([0, 40, 120]), np.array([10, 160, 255])))
                & table_mask)
        pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        W_img = hsv.shape[1]
        cands = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            a = cv2.contourArea(c)
            if 8 < a < 400 and 0.30 * W_img < x + w / 2 < 0.72 * W_img:
                # not the red block itself
                if "red_block" in out and abs(x + w / 2 - out["red_block"][0]) < 25:
                    continue
                cands.append((a, x + w / 2, y + h, _rect_angle(c), w, h))
        if cands:
            cands.sort(reverse=True)
            out["red_marker"] = (cands[0][1], cands[0][2], cands[0][3], cands[0][4], cands[0][5])
    if "red_marker" not in out:
        # marker fallback 2: white elongated body lying flat on the table
        white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 45, 255])) & table_mask
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        W_img = table_mask.shape[1]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            a = cv2.contourArea(c)
            # central table band only — excludes the teleop station and the robot arm
            if 60 < a < 2500 and w / max(1, h) > 1.6 and w < 120 \
                    and 0.30 * W_img < x + w / 2 < 0.72 * W_img:
                cands.append((a, x + w / 2, y + h, _rect_angle(c), w, h))
        if cands:
            cands.sort(reverse=True)
            out["red_marker"] = (cands[0][1], cands[0][2], cands[0][3], cands[0][4], cands[0][5])
    return out


def camera_pose_from_fk(gt0):
    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    urdf = os.environ.get("VEGA_URDF")
    robot = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=True, pos=(0, 0, 0)))
    scene.build()
    names = ([f"L_arm_j{i}" for i in range(1, 8)] + [f"R_arm_j{i}" for i in range(1, 8)]
             + ["torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3",
                "L_gripper_j1", "R_gripper_j1"])
    j2dof = {j.name: int(j.dofs_idx_local[0]) for j in robot.joints
             if getattr(j, "dofs_idx_local", None) is not None and len(j.dofs_idx_local) > 0}
    q = np.zeros(robot.n_dofs)
    for i, n in enumerate(names):
        q[j2dof[n]] = float(gt0[i])
    robot.set_dofs_position(q)
    scene.step()
    # genesis merges the massless zed links -> compose head_l3 pose with the fixed
    # URDF mount: xyz(0.0365, 0.023, 0.0489) rpy(-pi/2, 0, -pi/2)  (optical frame)
    link = robot.get_link("head_l3")
    pos = np.asarray(link.get_pos()).reshape(3)
    qw, qx, qy, qz = np.asarray(link.get_quat()).reshape(4)
    Rh = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    def rpy_mat(r, p, y_):
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y_), np.sin(y_)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    Rm = rpy_mat(-1.57079, 0.0, -1.57079)
    tm = np.array([0.0365, 0.023, 0.0489])
    cam_p = pos + Rh @ tm
    cam_R = Rh @ Rm
    return cam_p, cam_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="first-frame png (head_left)")
    ap.add_argument("--npz", required=True, help="gt/pred npz (gt[0] gives frame-0 joints)")
    ap.add_argument("--out", required=True, help="output json path")
    # camera_pose_from_fk already folds the URDF mount rpy(-pi/2, 0, -pi/2) into cam_R,
    # so cam_R IS the optical frame -> the opencv ray is used as-is. Passing "ros"
    # re-applies the convention a second time: rays fire backwards, every blob lands
    # metres behind the robot and gets dropped by the t <= 0 test below.
    ap.add_argument("--optical", default="opencv", choices=["ros", "opencv"],
                    help="ray convention for cam_R: opencv = optical (correct for this FK); "
                         "ros = extra x-fwd/z-up remap, only if cam_R is ever a body frame")
    args = ap.parse_args()

    bgr = cv2.imread(args.frame)
    h, w = bgr.shape[:2]
    table_mask = np.zeros((h, w), np.uint8)
    table_mask[int(h * 0.60):int(h * 0.86), :] = 255   # table surface band (excludes front edge)
    blobs = detect_blobs(bgr, table_mask)

    # standing marker: red-ish pixels extend well above the table contact point
    standing = {}
    if "red_marker" in blobs:
        u, v = int(blobs["red_marker"][0]), int(blobs["red_marker"][1])
        hsv_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        strip = hsv_full[max(0, v - 55):v + 4, max(0, u - 14):u + 14]
        reddish = ((cv2.inRange(strip, np.array([0, 40, 90]), np.array([12, 255, 255]))
                    | cv2.inRange(strip, np.array([160, 40, 90]), np.array([180, 255, 255]))) > 0)
        rows = np.where(reddish.any(axis=1))[0]
        extent = (rows.max() - rows.min() + 1) if len(rows) else 0
        standing["red_marker"] = bool(extent > 28)
        blobs["red_marker"] = blobs["red_marker"][:3]

    gt0 = np.load(args.npz)["gt"][0]
    cam_p, cam_R = camera_pose_from_fk(gt0)

    objs = {}
    for name, blob in blobs.items():
        u, v, ang_img = blob[0], blob[1], blob[2]
        d_cam = np.array([(u - CX) / FX, (v - CY) / FY, 1.0])   # opencv optical ray
        if args.optical == "ros":
            # ros body frame: x fwd, y left, z up  <-  opencv optical: z fwd, x right, y down
            d_cam = np.array([d_cam[2], -d_cam[0], -d_cam[1]])
        d_w = cam_R @ d_cam
        if abs(d_w[2]) < 1e-6:
            continue
        t = (TABLE_TOP_Z - cam_p[2]) / d_w[2]
        if t <= 0:
            continue
        p = cam_p + t * d_w
        # image-plane major axis -> world yaw (projected through the camera rotation)
        th = np.deg2rad(ang_img)
        d_img = np.array([np.cos(th), np.sin(th), 0.0])   # direction in optical frame (u, v)
        dw = cam_R @ d_img
        yaw = float(np.degrees(np.arctan2(dw[1], dw[0])))
        objs[name] = {"pos": [round(float(p[0]), 3), round(float(p[1]), 3)],
                      "yaw": round(yaw, 1), "standing": bool(standing.get(name, False))}

    json.dump(objs, open(args.out, "w"), indent=1)
    print("camera:", np.round(cam_p, 3), "objects:", objs)

    vis = bgr.copy()
    for name, blob in blobs.items():
        u, v = blob[0], blob[1]
        cv2.circle(vis, (int(u), int(v)), 6, DRAW[name], 2)
        cv2.putText(vis, name, (int(u) - 30, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, DRAW[name], 1, cv2.LINE_AA)
    cv2.imwrite(args.out.replace(".json", "_overlay.png"), vis)


if __name__ == "__main__":
    main()
