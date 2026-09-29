import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from common import *
res={}
m=load("real-s1")
s=S[:24]
z=m.answer(s).numpy()
# published reproduction: seed1 val logits
Sva=torch.tensor(ant(BT.harm("val")[0][:30]))
pub=np.load(TR/"seed1.npz")["real_trained"][:30]
rep=m.answer(Sva).numpy()
res["seed1_repro_maxdiff"]=float(np.abs(rep-pub).max())
# threads
for th in (1,4,8):
    torch.set_num_threads(th); res[f"threads{th}_equal"]=bool(np.array_equal(m.answer(s).numpy(),z))
torch.set_num_threads(2)
# batch run vs answer
with torch.no_grad():
    zb=m.run(s)[0].numpy()
res["batch_vs_answer_maxdiff"]=float(np.abs(zb-z).max()); res["batch_vs_answer_flips"]=int(((zb>=0)!=(z>=0)).sum())
# record=True parity
with torch.no_grad():
    lr,_,trace=m.run(s[:1],record=True)
res["record_true_vs_answer_equal"]=bool(float(lr[0])==float(z[0])); res["trace_dtype"]=str(trace.dtype)
ap,av=m.read_groups["dn"]
d16=(trace[-8:,ap].float().mean(1)-trace[-8:,av].float().mean(1)).mean(0)
lt=float(torch.exp(m.log_k)*d16*10+m.c)
res["trace_f16_reconstructed_logit_err"]=abs(lt-float(z[0]))
# thawed (training path) vs frozen
m.thaw()
with torch.no_grad(): zt=torch.cat([m.run(s[i:i+1])[0] for i in range(len(s))]).numpy()
res["thawed_vs_frozen_maxdiff"]=float(np.abs(zt-z).max())
m.freeze()
# freeze staleness: change kp_logm without freeze
with torch.no_grad(): saved=m.kp_logm.clone(); m.kp_logm.zero_()
zs=m.answer(s).numpy(); res["stale_after_kp_reset_equal_to_trained"]=bool(np.array_equal(zs,z))
m.freeze(); zr=m.answer(s).numpy(); res["after_refreeze_maxdiff"]=float(np.abs(zr-z).max())
with torch.no_grad(): m.kp_logm.copy_(saved)
m.freeze()
# rest: stale r_rest after memory change (no resettle)
with torch.no_grad(): m.kp_logm.zero_()
m.freeze(); zn=m.answer(s).numpy()
m.r_rest=None; m.settle(rest); zn2=m.answer(s).numpy()
res["reset_memory_stale_rest_vs_resettled_maxdiff"]=float(np.abs(zn-zn2).max())
with torch.no_grad(): m.kp_logm.copy_(saved)
m.freeze(); m.r_rest=None; st=m.settle(rest); res["settle_info"]=st
# settle continue from saved: rest fixed point and long run
r0=m.r_rest.clone()
with torch.no_grad():
    m.steps=1; _,r1,_=m.run(rest[None]); m.steps=80
    res["rest_one_step_max_change"]=float((r1[:,0]-r0).abs().max())
    m.steps=2000; _,rL,_=m.run(rest[None]); m.steps=80
    res["rest_2000_steps_max_drift"]=float((rL[:,0]-r0).abs().max())
    m.steps=40; _,_,tr=m.run(rest[None],record=True); m.steps=80
    dd=(tr[1:].float()-tr[:-1].float()).abs().amax(dim=(1,2))
    res["rest_trace_step_change_max"]=float(dd.max())
    # alternative rest: settle from a different initial state (0.5 everywhere)
    m.r_rest=None; m.steps=80
    rr=torch.full((m.n,),0.3); 
    for _ in range(50):
        m.steps=40; _,rr2,_=m.run(rest[None],r0=rr); rr=rr2[:,0]
    m.steps=80
    res["rest_from_0.3_vs_from_0_max"]=float((rr-r0).abs().max())
    m.r_rest=rr; za=m.answer(s).numpy(); res["alt_rest_answer_maxdiff"]=float(np.abs(za-z).max()); res["alt_rest_flips"]=int(((za>=0)!=(z>=0)).sum())
    m.r_rest=r0
# saved start r_rest vs resettled (untrained)
m0=load(None)
rs=m0.r_rest.clone(); z0=m0.answer(s).numpy(); m0.r_rest=None; info=m0.settle(rest); z0b=m0.answer(s).numpy()
res["start_saved_rest_vs_resettled_state_max"]=float((m0.r_rest-rs).abs().max()); res["start_saved_vs_resettled_logit_max"]=float(np.abs(z0-z0b).max()); res["start_resettle"]=info
# float64
m64=load("real-s1"); m64.double(); m64.r_rest=None; torch.set_default_dtype(torch.float64)
with torch.no_grad():
    m64.freeze(); m64.settle(rest.double()); z64=m64.answer(s.double()).numpy()
res["f64_vs_f32_maxdiff"]=float(np.abs(z64-z).max()); res["f64_flips"]=int(((z64>=0)!=(z>=0)).sum())
print(json.dumps(res,indent=1,default=str))
json.dump(res,open(AUD/"a4.json","w"),indent=1,default=str)
