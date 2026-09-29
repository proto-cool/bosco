import sys,json
for f in sys.argv[1:]:
    print('##',f)
    for G,r in json.load(open(f)).items():
        c=r['calib']; print('G',G,'kc_act %.3f kc_rate %.3f mbon %.3f'%(c['final']['kc_active'],c['final']['kc_rate'],c['final']['mbon']),'settle',c['settle']['converged'],'%.2g'%c['settle']['max_step_change'])
        for nm in ('rest','smell','rest_from0','rest_from_sat'):
            e=r[nm]; print('   %-13s step %.2g osc %d dead %.3f sat %.4f mean %.4f'%(nm,e['max_step_change_last200'],e['n_osc_cells'],e['share_dead(<1e-3)'],e['share_sat(>0.9)'],e['mean_rate']),
              ('rho %.4f ReM %.3f ImM %.3f %s'%(e['rho'],e['maxRe_M'],e['imM'],e['lead_types'][:3])) if 'rho' in e else ('vs_rest %.3f share %.4f'%(e['maxabs_vs_rest'],e['share_diff']) if 'maxabs_vs_rest' in e else ''), e['osc_types'][:4] if e['n_osc_cells'] else '')
