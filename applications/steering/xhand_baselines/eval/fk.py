"""Vega-1 forward kinematics for the arm metrics (pinocchio, robot model from the T-Rex quickstart).

Reduced bimanual Vega-1 (arms only; torso/head frozen at capture defaults). The configuration is
[L_arm_j1..j7, R_arm_j1..j7], the order of the dataset's arm slices, so the 14-D arm vector is qpos.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.environ["TREX_DIR"], "dataset_quickstart", "src"))
import pinocchio as pin  # noqa: E402
from trex_dataset_quickstart import robot as qr  # noqa: E402
from trex_dataset_quickstart import schema as qs  # noqa: E402


def build_fk():
    rw, _assemble, _disassemble = qr.build_reduced_bimanual_robot(
        {"head": qr.DEFAULT_HEAD, "torso": qr.DEFAULT_TORSO}
    )
    model, data = rw.model, rw.data
    lid, rid = model.getFrameId("L_ee"), model.getFrameId("R_ee")
    assert model.nq == 14, f"expected 14 arm DoF, got {model.nq}"
    expected = list(qs.LEFT_ARM_JOINTS) + list(qs.RIGHT_ARM_JOINTS)
    assert list(model.names)[1:] == expected, (list(model.names)[1:], expected)

    def fk(arm14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[N,14] arm joint positions -> ([N,4,4] L_ee, [N,4,4] R_ee)."""
        n = arm14.shape[0]
        L = np.empty((n, 4, 4)); R = np.empty((n, 4, 4))
        for i in range(n):
            q = np.asarray(arm14[i], dtype=np.float64)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            L[i] = data.oMf[lid].homogeneous
            R[i] = data.oMf[rid].homogeneous
        return L, R

    return fk, model.nq
