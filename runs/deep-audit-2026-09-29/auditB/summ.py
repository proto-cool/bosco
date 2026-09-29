import sys,json
for f in sys.argv[1:]:
    print('##', f)
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        print(r['beta'],r['k'],[(nm,round(r[nm]['max_step_change_last200'],5),r[nm]['n_osc_cells'],round(r[nm].get('rho',0),4),round(r[nm].get('maxRe_M',0),3),round(r[nm].get('imM',0),2),round(r[nm]['read'],3),round(r[nm]['share_sat'],4),round(r[nm]['share_dead'],3),r[nm].get('lead_types',[])[:3]) for nm in ('rest','smell')])
