"""Train ViTacFormer on an XHand LeRobot v3 dataset (step-based version of imitate_episodes.train_bc).

Same model, losses (L1 action + kl_weight * KL + L1 future tactile), AdamW param groups, cosine schedule
with 1000 linear warmup steps, and hard-switch future-tactile curriculum (ground truth to the action
decoder for the first --tf-steps, predicted afterwards). Multi-GPU: torchrun (DDP); --batch-size is the total.

  torchrun --nproc_per_node N train_xhand.py --dataset <lerobot root> --episodes <train.json> --out <dir>
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.distributed as dist

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "detr")]
try: import IPython  # noqa: F401  (imported, unused, by the model files)
except ImportError: import types; sys.modules["IPython"] = types.SimpleNamespace(embed=None)
from dataset.xhand_lerobot import XHandLeRobot, CAMS, HIST, TAC_HIST, worker_init


def policy_config(a, state_dim, force_dim):
    return {"lr": a.lr, "num_queries": a.chunk, "kl_weight": a.kl_weight, "hidden_dim": 512, "dim_feedforward": 3200,
            "lr_backbone": 1e-5, "backbone": "resnet18", "enc_layers": 4, "dec_layers": 7, "nheads": 8,
            "camera_names": CAMS, "use_tactile": a.tactile, "action_dim": state_dim,
            "qpos_dim": HIST * state_dim, "tactile_dim": 2 * force_dim, "tactile_steps": TAC_HIST}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--episodes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--kl-weight", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=100); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--tf-steps", type=int, default=None, help="default: 75/2000 of --steps (original: 75 of 2000 epochs)")
    ap.add_argument("--no-tactile", dest="tactile", action="store_false")
    ap.add_argument("--save-freq", type=int, default=5000); ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tf_steps = round(a.steps * 75 / 2000) if a.tf_steps is None else a.tf_steps
    if not a.tactile: tf_steps = 0

    ddp, gpu = "RANK" in os.environ, a.device.startswith("cuda")
    if ddp:                                   # gloo on CPU only for testing
        dist.init_process_group("nccl" if gpu else "gloo")
        if gpu: torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    dev = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if ddp and gpu else a.device)
    assert a.batch_size % world == 0, f"batch {a.batch_size} not divisible by {world} processes"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    ds = XHandLeRobot(a.dataset, json.load(open(a.episodes)), train=True, tactile=a.tactile)
    sd, fd = ds.low[ds.eps[0]]["state"].shape[1], ds.low[ds.eps[0]]["force"].shape[1]
    cfg = policy_config(a, sd, fd)
    from policy import ACTPolicy
    policy = ACTPolicy(cfg).to(dev)
    opt = policy.configure_optimizers()
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=a.warmup, num_training_steps=a.steps)
    model = torch.nn.parallel.DistributedDataParallel(policy, find_unused_parameters=True) if ddp else policy  # is_pad_head is unused

    sampler = torch.utils.data.DistributedSampler(ds, world, rank, shuffle=True, seed=a.seed, drop_last=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size // world, sampler=sampler, num_workers=a.num_workers,
                                     drop_last=True, worker_init_fn=worker_init, persistent_workers=a.num_workers > 0,
                                     prefetch_factor=4 if a.num_workers else None)
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        print(f"[vitacformer] episodes={len(ds.eps)} samples={len(ds)} tactile={a.tactile} steps={a.steps} batch={a.batch_size} "
              f"({world} x {a.batch_size // world}) tf_steps={tf_steps} lr={a.lr} out={a.out}", flush=True)

    def save(step):
        if rank: return
        torch.save({"model": policy.state_dict(), "policy_config": cfg, "stats": ds.stats, "step": step,
                    "tactile": a.tactile, "args": vars(a)},
                   f"{a.out}/policy_last.ckpt" if step == a.steps else f"{a.out}/policy_step{step:06d}.ckpt")

    step, epoch, t0, hist, cnt = 0, 0, time.time(), {}, 0
    policy.train()
    while step < a.steps:
        sampler.set_epoch(epoch); epoch += 1
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            gt_tac = step < tf_steps
            if a.tactile:
                ld = model(b["qpos"], b["image"], b["action"], b["is_pad"], dev, b["tactile"], b["tactile_next"], use_gt_tactile=gt_tac)
            else:
                ld = model(b["qpos"], b["image"], b["action"], b["is_pad"], dev)
            ld["loss"].backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            for k, v in ld.items(): hist[k] = hist.get(k, 0.0) + float(v)
            cnt += 1
            if rank == 0 and (step % 100 == 0 or step == a.steps or step == 1):
                print(f"step {step}/{a.steps} " + " ".join(f"{k}={v / cnt:.4f}" for k, v in hist.items()) +
                      f" lr={sched.get_last_lr()[0]:.2e}" + (f" tactile_in={'gt' if gt_tac else 'pred'}" if a.tactile else "") + f" {time.time() - t0:.0f}s", flush=True)
                hist, cnt = {}, 0
            if step % a.save_freq == 0 or step == a.steps: save(step)
            if step >= a.steps: break
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
