"""Final stages: SQP polish and the KKT finisher.

Algorithm 2 stops on a cost plateau, and its fixed point need not be stationary because Algorithm 1
restores feasibility by merit descent rather than by the exact Euclidean projection.  The polish takes
reduced Newton steps on the Lagrangian (a small QP with the linearized near-contacts), and the finisher
solves the KKT system with the identified contacts by an active-set Newton method, which puts every
active block exactly on the boundary of its cone and gives the multipliers the certificate needs."""
import time
from math import comb

import numpy as np
import torch

from .certificate import _nnls_safe
from .spline import DTYPE as DT


class PolishMixin:
    def polish(self,Z,tv,iters=60,deadline=None,verbose=False):
        """SQP polish: reduced Newton on the Lagrangian as a small QP with the linearized
        near-contacts as inequalities (no penetration to first order) and a box trust region;
        retraction restores active contacts to their pre-step depth; Alg 1 restores tol;
        accept on cost decrease; trust radius adapts. Exact Hessian (blocks are quadratic)."""
        import cvxpy as cp
        eta=1e-3; a=1.0; nfail=0
        for it in range(iters):
            if deadline and time.time()>deadline: break
            cts=self.contacts(Z,thresh=5e-2)
            gc=self.cost_grad(Z).reshape(-1)
            if len(cts)==0:
                Gm=torch.zeros(gc.numel(),0,dtype=DT); w=np.zeros(0); v0=[]
            else:
                vv,Gm=self.pvals_grads(Z,cts); v0=vv.tolist()
                w=_nnls_safe(Gm.cpu().numpy(),gc.cpu().numpy())
            bds=[self.atom(s_) for _,s_ in cts]
            def Lf(z):
                Zz=z.view(self.n,self.d_f); pi=self.pi_all(Zz)
                return self.cost(Zz)-sum(float(w[i])*(pi[cts[i][0]]*bds[i]).sum() for i in range(len(cts)))
            HL=torch.autograd.functional.hessian(Lf,Z.reshape(-1).clone()); HL=0.5*(HL+HL.T)
            ev,V=torch.linalg.eigh(HL); evmax=float(ev.max().clamp_min(1e-12)); Hp=(V*ev.clamp_min(1e-6*evmax))@V.T
            Hn=Hp.cpu().numpy(); Hn=0.5*(Hn+Hn.T)
            c0=float(self.cost(Z)); acc=False
            # one QP: min 1/2 x'Hp x + gc'x  s.t. linearized near-contacts G'x >= -max(v0,0)  (scaling x by a<1 keeps it feasible)
            x=cp.Variable(gc.numel()); cons=[]
            if len(cts)>0:
                Gn=Gm.cpu().numpy().T; rhs=np.array([-max(v,0.0) for v in v0]); cons.append(Gn@x>=rhs)
            pr=cp.Problem(cp.Minimize(0.5*cp.quad_form(x,cp.psd_wrap(Hn))+gc.cpu().numpy()@x),cons)
            try: pr.solve(solver=cp.CLARABEL)
            except Exception as e: x.value=None
            if x.value is None or pr.status not in ('optimal','optimal_inaccurate'):
                # fallback: steepest descent in the tangent space of the active contacts
                if verbose: print(f"      it{it} QP status {pr.status}: tangent-gradient fallback",flush=True)
                act=[i for i in range(len(cts)) if w[i]>0 or v0[i]<=0]
                if len(act)>0:
                    Ga=Gm[:,act]; U,Sv,Vh=torch.linalg.svd(Ga.T,full_matrices=True); rk=int((Sv>1e-8*Sv.max()).sum()); T=Vh[rk:].T
                    dzf=-(T@(T.T@gc))
                else: dzf=-gc
                x.value=(dzf/dzf.norm().clamp_min(1e-300)*0.01*max(1.0,float(Z.norm()))).cpu().numpy()
            dz0=torch.as_tensor(x.value,dtype=DT); a=1.0
            for trial in range(10):
                dz=a*dz0
                Zt=(Z.reshape(-1)+dz).view(self.n,self.d_f)
                for _ in range(10):
                    if len(cts)==0: break
                    vals,Gt=self.pvals_grads(Zt,cts)
                    r=vals-torch.tensor([min(v,0.0)*0.7071 for v in v0],dtype=DT)
                    r=torch.clamp(r,max=0.0)
                    if float(r.abs().max())<1e-12: break
                    # cost-aware (Hessian-weighted) least-norm retraction: min 1/2 D'Hp D s.t. Gt' D = -r
                    HiG=V@((V.T@Gt)/ev.clamp_min(1e-6*evmax).view(-1,1))
                    Zt=(Zt.reshape(-1)-HiG@torch.linalg.pinv(Gt.T@HiG,rtol=1e-8)@r).view(self.n,self.d_f)
                Zt,eta,ok=self.project(Zt,tv,eta,cap=400,hyst=0.5)
                ct=float(self.cost(Zt))
                if verbose: print(f"      it{it} a={a:.3e} |dz|={float(dz.norm()):.3e} ok={ok} cost {ct:.4f} (c0 {c0:.4f})",flush=True)
                if ok and ct<c0-1e-12*abs(c0):
                    Z=Zt; acc=True; a=min(1.0,2*a); break
                a*=0.25
            if verbose: print(f"   polish {it}: cost {float(self.cost(Z)):.6f} contacts {len(cts)} acc={acc} a {a:.2e}",flush=True)
            if not acc:
                nfail+=1
                if nfail>=2: break
            else:
                nfail=0
                if (c0-float(self.cost(Z)))<1e-10*abs(c0): break
        return Z,eta

    def kkt_finish(self, Z, **kw):
        return kkt_finish(self, Z, **kw)


# ---------------------------------------------------------------- KKT finisher
def _b2d_torch(d2,s):
    """differentiable Bernstein basis of degree d2 at scalar tensor s"""
    ks=torch.arange(d2+1,dtype=DT); cb=torch.tensor([comb(d2,k) for k in range(d2+1)],dtype=DT)
    return cb*s**ks*(1-s)**(d2-ks)
def _kkt_finish_once(p,Z,cts=None,iters=30,verbose=False,thresh=5e-2,prune=True,athresh=None,merge=True,boxes=None):
    """Active-set Newton on the KKT system of (P) with the identified contacts:
       grad cost = sum_i lam_i grad_Z p_i(Z,s_i);  p_i(Z,s_i)=0;  d/ds p_i(Z,s_i)=0  (double root on the
       inflated obstacle => block exactly in K, dist 0).  Unknowns (Z, s, lam). Warm start from the
       polished point. Returns Z, contacts, lam, and a report (stationarity, feasibility, eigen test)."""
    d2=2*p.cd
    if cts is None: cts=list(p.contacts(Z,thresh=thresh,athresh=athresh))
    # drop contacts sitting on a pinned endpoint (value constant in Z) and any contact with a vanishing gradient
    def _pinned(c):
        b,s_=c
        if isinstance(s_,tuple): return False
        seg=p.types[b][2] if len(p.types[b])>2 else None
        return (seg==0 and s_<1e-6) or (seg==p.M-1 and s_>1-1e-6)
    cts=[c for c in cts if not _pinned(c)]
    if len(cts)==0: cts=[]
    vv,Gm=p.pvals_grads(Z,cts) if len(cts) else (torch.zeros(0,dtype=DT),torch.zeros(p.n*p.d_f,0,dtype=DT)); gc=p.cost_grad(Z).reshape(-1)
    if len(cts):
        gnrm=Gm.norm(dim=0); okg=(gnrm>1e-6*float(gnrm.max())).tolist()
        cts=[c for c,o in zip(cts,okg) if o]; vv,Gm=p.pvals_grads(Z,cts) if len(cts) else (torch.zeros(0,dtype=DT),torch.zeros(p.n*p.d_f,0,dtype=DT))
    w=_nnls_safe(Gm.cpu().numpy(),gc.cpu().numpy())
    wmax=float(w.max()) if len(w) else 0.0
    keep=[i for i in range(len(cts)) if (not prune) or (w[i]>1e-2*wmax) or (vv[i]<=0 and w[i]>1e-12)]
    # merge near-duplicate contacts on the same block (|ds|<0.1): keep the one with the larger weight
    keep2=[]
    def _same(i,j):
        # duplicates: same block index within |ds|<0.1, or the same obstacle on adjacent segments within 0.15 of
        # the global parameter (a junction twin: end of one segment / start of the next)
        if isinstance(cts[i][1],tuple) or isinstance(cts[j][1],tuple): return cts[j][0]==cts[i][0] and cts[j][1]==cts[i][1]
        if cts[j][0]==cts[i][0]: return abs(cts[j][1]-cts[i][1])<0.1
        ti,tj=p.types[cts[i][0]],p.types[cts[j][0]]
        if ti[:2]!=tj[:2] or abs(ti[2]-tj[2])!=1: return False
        bi=cts[i][1]<0.02 or cts[i][1]>0.98; bj=cts[j][1]<0.02 or cts[j][1]>0.98
        return (bi or bj) or abs((ti[2]+cts[i][1])-(tj[2]+cts[j][1]))<0.15
    def _bnd(i): return (not isinstance(cts[i][1],tuple)) and (cts[i][1]<0.02 or cts[i][1]>0.98)
    for i in sorted(keep,key=lambda i:(_bnd(i),-w[i])):     # interior contacts first, then by weight
        if merge and any(_same(i,j) for j in keep2): continue
        keep2.append(i)
    keep=sorted(keep2)
    cts=[cts[i] for i in keep]; w=w[keep]
    nz=p.n*p.d_f; mc=len(cts); idx=torch.tensor([b for b,_ in cts])
    if mc==0:
        # no active contact: the point is a KKT point iff the cost gradient vanishes on the chart; then the
        # unconstrained minimizer is feasible and the certificate is trivial (Lagrangian = cost, convex)
        gn=float(p.cost_grad(Z).norm()); pi=p.pi_all(Z); dd,_=p.resid(pi); feas=bool((dd<=p.tolvec()).all())
        return Z,dict(contacts=[],lam=[],F=gn,F_rel=gn/max(1.0,gn),max_d2=float(dd.max()),cost=float(p.cost(Z)),true_margins=p.true_clearance(Z),lam_nonneg=True,lam_min=float(2*torch.linalg.eigvalsh(p.H).min()),lam_max=float(2*torch.linalg.eigvalsh(p.H).max()),lam_min_rel=float(torch.linalg.eigvalsh(p.H).min()/torch.linalg.eigvalsh(p.H).max()),kkt_point=bool(feas and gn<1e-6*max(1.0,float(p.cost(Z)))),certified=bool(feas and gn<1e-6*max(1.0,float(p.cost(Z)))),unconstrained=True)
    is_row=[isinstance(s_,tuple) for _,s_ in cts]
    s0=torch.tensor([0.0 if isinstance(s_,tuple) else s_ for _,s_ in cts],dtype=DT); fixed_s=(s0<1e-6)|(s0>1-1e-6)
    if p.mode=='sampled': fixed_s=torch.ones(mc,dtype=torch.bool)      # atoms are the fixed sample points
    fixed_s=fixed_s|torch.tensor(is_row,dtype=torch.bool)
    rowA=[p.Eo[s_[1]] if isinstance(s_,tuple) else None for _,s_ in cts]
    # A contact pinned at a segment end (s = 0 or 1, e.g. the knot where a half-space window starts) keeps a
    # CONSTANT basis row: differentiating the Bernstein basis through s at the boundary yields non-finite
    # Jacobian entries, which made the Newton step abort at iteration 0 on the box family.
    for i in range(mc):
        if rowA[i] is None and bool(fixed_s[i]): rowA[i]=_b2d_torch(d2,s0[i]).detach()
    x=torch.cat([Z.reshape(-1).clone(),s0.clone(),torch.as_tensor(w,dtype=DT)])
    def F(x):
        z=x[:nz]; s=x[nz:nz+mc]; lam=x[nz+mc:]
        with torch.enable_grad():
            if not x.requires_grad: x=x.detach().requires_grad_(True); z=x[:nz]; s=x[nz:nz+mc]; lam=x[nz+mc:]
            pi=p.pi_all(z.view(p.n,p.d_f))
            vals=torch.stack([(pi[idx[i]]*(rowA[i] if rowA[i] is not None else _b2d_torch(d2,s[i]))).sum() for i in range(mc)])
            L=p.cost(z.view(p.n,p.d_f))-(lam*vals).sum()
            gL,=torch.autograd.grad(L,[z],create_graph=True)
            if bool((~fixed_s).any()):
                ds,=torch.autograd.grad(vals.sum(),[s],create_graph=True,allow_unused=True)
                ds=s-s0 if ds is None else torch.where(fixed_s,s-s0,ds)   # pinned atoms: equation s - s0 = 0
            else: ds=s-s0
        return torch.cat([gL,vals,ds]),vals
    mu=1e-6; gcn0=max(1.0,float(gc.norm()))
    for it in range(iters):
        J=torch.autograd.functional.jacobian(lambda y: F(y)[0],x)
        Fx,_=F(x); Fx=Fx.detach(); nrm=float(Fx.norm())
        if verbose: print(f"   kkt {it}: |F|={nrm:.3e} cost {float(p.cost(x[:nz].view(p.n,p.d_f))):.6f} lam {x[nz+mc:].tolist()}",flush=True)
        if nrm<1e-11 or nrm<1e-9*gcn0: break      # stationarity is relative to the cost gradient; iterating past it buys nothing
        Jn=J.detach().cpu().numpy(); Fn=Fx.cpu().numpy()
        if not (np.isfinite(Jn).all() and np.isfinite(Fn).all()): break
        # Exact Newton step first (quadratic convergence when the Jacobian is nonsingular), then Levenberg-Marquardt
        # damping as a fallback: (J^T J + mu D) dx = -J^T F with Marquardt scaling D = diag(J^T J).  Plain least
        # squares alone blows up along near-null directions when several contacts share a segment.
        accepted=False
        import scipy.linalg as sla
        try:
            dxn=sla.lstsq(Jn,-Fn,lapack_driver='gelsd')[0]; a=1.0
            for _ in range(24):
                xt=x+a*torch.as_tensor(dxn,dtype=DT); xt[nz:nz+mc]=xt[nz:nz+mc].clamp(0,1)
                if float(F(xt)[0].detach().norm())<nrm: x=xt.detach(); accepted=True; break
                a*=0.5
        except Exception: pass
        if accepted: continue
        JtJ=Jn.T@Jn; JtF=Jn.T@Fn; Dg=np.clip(np.diag(JtJ),1e-12,None)
        for _ in range(12):
            try: dx=np.linalg.solve(JtJ+mu*np.diag(Dg),-JtF)
            except np.linalg.LinAlgError: mu*=10; continue
            xt=x+torch.as_tensor(dx,dtype=DT); xt[nz:nz+mc]=xt[nz:nz+mc].clamp(0,1)
            if float(F(xt)[0].detach().norm())<nrm: x=xt.detach(); mu=max(mu/3,1e-15); accepted=True; break
            mu*=10
        if not accepted: break
    z=x[:nz]; s=x[nz:nz+mc]; lam=x[nz+mc:]; Zf=z.view(p.n,p.d_f)
    pi=p.pi_all(Zf); dd,_=p.resid(pi)
    rep=dict(_cts=[(b,(cts[i][1] if rowA[i] is not None else float(s[i]))) for i,(b,_) in enumerate(cts)],contacts=[(p.types[b],(cts[i][1] if rowA[i] is not None else float(s[i]))) for i,(b,_) in enumerate(cts)],lam=lam.tolist(),F=float(F(x)[0].detach().norm()),
             max_d2=float(dd.max()),cost=float(p.cost(Zf)),true_margins=p.true_clearance(Zf),lam_nonneg=bool((lam>=-1e-12).all()))
    # certificate with these exact atoms/multipliers: Hessian of L on the chart
    A=[(rowA[i] if rowA[i] is not None else _b2d_torch(d2,s[i]).detach()) for i in range(mc)]
    def Lf(zz):
        Zz=zz.view(p.n,p.d_f); pi=p.pi_all(Zz); return p.cost(Zz)-sum(float(lam[i])*(pi[idx[i]]*A[i]).sum() for i in range(mc))
    HL=torch.autograd.functional.hessian(Lf,z.detach().clone())
    nq=0
    if p.qrt and boxes is not None:
        HL,nq=p.qrt_bound(Zf,[(int(idx[i]),(cts[i][1] if rowA[i] is not None else float(s[i])),float(lam[i])) for i in range(mc)],boxes,HL)
    ev=torch.linalg.eigvalsh(0.5*(HL+HL.T))
    rep.update(lam_min=float(ev.min()),lam_max=float(ev.max()),n_quartic=nq,
               region=('corridor' if (p.qrt and boxes is not None) else 'global'))
    gcn=float(p.cost_grad(Zf).norm()); rep['F_rel']=rep['F']/max(1.0,gcn); rep['lam_min_rel']=rep['lam_min']/rep['lam_max']
    rep['kkt_point']=bool(rep['F_rel']<1e-6 and rep['lam_nonneg'] and rep['max_d2']<=1e-12*max(1.0,float(pi.abs().max())**2))
    # With a quartic block the Hessian varies with Gamma, so the eigenvalue test means global optimality
    # only when it was run against a region bound; without one it establishes second-order local optimality.
    rep['second_order']=bool(rep['kkt_point'] and rep['lam_min']>=-1e-9*rep['lam_max'])
    rep['certified']=bool(rep['second_order'] and (boxes is not None or not p.qrt))
    return Zf,rep


def kkt_finish(p,Z,cts=None,iters=30,verbose=False,thresh=5e-2,passes=8,athresh=None,merge_first=True,boxes=None):
    """Outer active-set loop around the Newton finisher.  (i) A contact whose multiplier converges negative is
       not active at the local minimizer (relaxing it lowers the cost): drop it and re-solve.  (ii) A block that
       the Newton step drove out of K was missing from the active set (it had slack at the warm start): add the
       contacts detected at the new point and re-solve.  At most `passes` Newton runs; the last report is returned."""
    Zc=Z; rep=None; cts_c=cts; best=None
    for ps in range(passes):
        # merge junction twins only on the first pass; once the outer loop has explicitly re-added a block
        # that the Newton step left outside K, merging would delete it again and the loop would not progress
        Zn,rep=_kkt_finish_once(p,Zc,cts=cts_c,iters=iters,verbose=verbose,thresh=thresh,prune=(ps==0),athresh=athresh,merge=(ps==0 and merge_first),boxes=boxes)
        if rep.get('unconstrained') or rep['kkt_point']: return Zn,rep
        if best is None or (rep['F_rel']<best[1]['F_rel'] and rep['max_d2']<=best[1]['max_d2']): best=(Zn,rep)
        lam=np.array(rep['lam']) if len(rep['lam']) else np.zeros(0)
        neg=[i for i in range(len(lam)) if lam[i]< -1e-8*max(1.0,float(np.abs(lam).max()))]
        # The outer loop is chasing an EXACT KKT point, so a block counts as outside K at the threshold the
        # kkt_point test itself uses, not at the looser margin tolerance the algorithm runs to.
        pi=p.pi_all(Zn); dd,_=p.resid(pi)
        _kthr=1e-12*max(1.0,float(pi.abs().max())**2)
        viol=bool((dd>_kthr).any())
        if rep['F_rel']>5e-2 and not viol: return best      # Newton made no progress at all
        keep=[c for i,c in enumerate(rep['_cts']) if i not in neg]
        if viol:
            # blocks driven out of K: add every contact detected at the new point that is not yet in the set
            cand=[c for c in p.contacts(Zn,thresh=max(thresh,5e-2),athresh=athresh) if not any(c[0]==k[0] and (c[1]==k[1] if isinstance(c[1],tuple) or isinstance(k[1],tuple) else abs(c[1]-k[1])<0.1) for k in keep)]
            if cand:
                vv_,_=p.pvals_grads(Zn,cand); new=[c for c,v in zip(cand,vv_.tolist()) if v<0]    # only blocks actually out of K
            else: new=[]
            if not new: return best
            if verbose: print(f"   active-set pass {ps}: {len(new)} blocks out of K, adding {[(p.types[b],s_) for b,s_ in new][:6]}",flush=True)
            keep=keep+new
        elif neg:
            if verbose: print(f"   active-set pass {ps}: dropping {[rep['contacts'][i] for i in neg]}",flush=True)
        else: return best
        Zc=Zn; cts_c=keep
    return best if best is not None else (Zn,rep)
