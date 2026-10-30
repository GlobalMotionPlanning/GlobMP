"""DiRecT's Maze2D evaluator with GlobMP in place of IPOPT in DiRecT's MPC step.

Everything else is DiRecT's: checkpoint, sampler, fitted dynamics, PD execution, metrics, and the 'direct' policy config
(configs/policies/direct.yaml). The only change: policy id 'direct' is mapped to GlobMPDiRecTPolicy (globmp_mpc.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DIRECT = os.path.abspath(os.path.join(HERE, '..', 'maze2d_baselines', 'DiRecT'))
sys.path.insert(0, DIRECT); sys.path.insert(1, HERE)
os.chdir(DIRECT)
import torch
if os.environ.get('GMP_THREADS'):
    torch.set_num_threads(int(os.environ['GMP_THREADS']))
import policy
from globmp_mpc import GlobMPDiRecTPolicy
policy.POLICY_DICT['direct'] = GlobMPDiRecTPolicy
import eval as ev
ev.main()
