import json,sys
v=sys.argv[1]
for a in ['real','layered','hash']:
  try: d=json.load(open(f'A_{v}_{a}.json'))
  except Exception as e: print(a,e); continue
  e1=d['E1']; print('==',a)
  for s in ['items','synthetic']:
    x=e1[s]; print(' E1',s,'LS orn %.3f uniPN %.3f allPN %.3f | skew orn %.2f uniPN %.2f | mean ev orn %.3f uniPN %.3f'%(x['orn']['lifetime_sparseness'],x['uni_pn']['lifetime_sparseness'],x['all_pn']['lifetime_sparseness'],x['orn']['skew_pooled'],x['uni_pn']['skew_pooled'],x['orn']['mean_evoked'],x['uni_pn']['mean_evoked']))
  print(' E1 glom',e1['glom_specificity'],'orn-pn corr',round(e1['orn_pn_same_glom_corr_median'],3))
  e=d['E2']; print(' E2 kc %.3f->%.3f x%.2f J %.3f->%.3f'%(e['kc_frac_intact'],e['kc_frac_apl_off'],e['ratio'],e['jaccard_intact'],e['jaccard_apl_off']))
  e=d['E3']; print(' E3',{k:(round(v,3) if isinstance(v,float) else v) for k,v in e.items()})
  e=d['E4']; print(' E4 cut typ %.1f all %.1f sd %.1f; typ ev %.4f->%.4f'%(e['cut_pct_typical'],e['cut_pct_all'],e['cut_pct_sd'],e['typical_abs_evoked_intact'],e['typical_abs_evoked_kp_off']))
  e=d['E7']; print(' E7',{k:round(v,4) for k,v in e.items() if k!='pass'})
