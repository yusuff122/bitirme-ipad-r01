"""Create the experiment comparison report without inventing missing metrics."""
import argparse
import json
from pathlib import Path


BASE={
 'E0':{'AUROC':0.7282238589,'AUPR':0.4883859957,'F1':0.5856975381,'Precision':0.4148123547,'Recall':0.9960127592,'Accuracy':0.5204884668,'TP':1249,'FP':1762,'TN':669,'FN':5},
 'E1':{'AUROC':0.5932755208,'AUPR':0.3846233176,'F1':0.2723900371,'Precision':0.4060031596,'Recall':0.2049441786,'Accuracy':0.6274084125,'TP':257,'FP':376,'TN':2055,'FN':997},
 'E2':{'AUROC':0.6440241905,'AUPR':0.4351160638,'F1':0.3888372093,'Precision':0.4665178571,'Recall':0.3333333333,'Accuracy':0.6434192673,'TP':None,'FP':None,'TN':None,'FN':None},
 'E3':{'AUROC':None,'AUPR':None,'F1':None,'Precision':None,'Recall':None,'Accuracy':None,'TP':None,'FP':None,'TN':None,'FN':None}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--e3-metrics'); p.add_argument('--output-dir',default='outputs/reports'); a=p.parse_args(); rows={k:dict(v) for k,v in BASE.items()}
    if a.e3_metrics:
        m=json.loads(Path(a.e3_metrics).read_text()); rows['E3']={key:m.get(key.lower()) for key in rows['E3']}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'E0_E1_E2_E3.json').write_text(json.dumps(rows,indent=2))
    fields=list(next(iter(rows.values())).keys()); lines=['# E0–E3 karşılaştırması','', '| Deney | '+' | '.join(fields)+' |','|---|'+'|'.join(['---:']*len(fields))+'|']
    for name,metrics in rows.items(): lines.append('| '+name+' | '+' | '.join('—' if metrics[k] is None else f"{metrics[k]:.6f}" if isinstance(metrics[k],float) else str(metrics[k]) for k in fields)+' |')
    lines+=['','E3 boş alanları final değerlendirme yapılana kadar bilerek boş bırakılmıştır. E2 confusion counts kaynak sonuç metninde verilmediği için uydurulmamıştır.']
    (out/'E0_E1_E2_E3.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
