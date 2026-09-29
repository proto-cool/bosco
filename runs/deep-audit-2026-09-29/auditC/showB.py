import json,sys
v=sys.argv[1]
for a in sys.argv[2:]:
  try: d=json.load(open(f'B_{v}_{a}.json'))
  except Exception as e: print(a,e); continue
  print('==',a)
  e6=d['E6']; n=0; ok=0; thr=1e-4
  for t,x in e6.items():
    meas=abs(x['d_read'])>thr; n+=meas; ok+=meas and x['sign_ok']
    print(' E6 %-12s %s model=%-8s dM=%.3f dread=%+.2e %s'%(t,x['lit_nt'],x['model_group'],x['d_rate_mbon'],x['d_read'],'OK' if x['sign_ok'] else 'WRONG'))
  print(' E6 sign ok %d/%d measurable (|d|>1e-4)'%(ok,n))
  p=[0,0,0,0]
  for t,x in d['E5'].items():
    if 'drop_A_pct' not in x: print(' E5',t,'no response'); continue
    p[0]+=1;p[1]+=x['pass_dropA'];p[2]+=x['pass_specific'];p[3]+=x['pass_read_dir']
    print(' E5 %-12s evA %.4f dropA %5.1f dropB %5.1f J %.3f dreadA %+.2e dreadB %+.2e %s'%(t,x['evoked_A_before'],x['drop_A_pct'],x['drop_B_pct'],x['jaccard_AB'],x['d_read_A'],x['d_read_B'],'dir OK' if x['pass_read_dir'] else 'dir WRONG'))
  print(' E5 n=%d dropA>=50: %d specific: %d read dir: %d'%tuple(p))
  e8=d['E8']; print(' E8', json.dumps({k:({kk:round(vv,4) if isinstance(vv,float) else vv for kk,vv in x.items()} if isinstance(x,dict) else x) for k,x in e8.items()}))
