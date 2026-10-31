"""Run authorized frozen clinical inference, then verified table refill.

Foreground supervisor keeps subprocesses alive in a managed long-running session.
On an inference error, stop and leave the manuscript unfilled; no partial result
is silently published. Original partial-synthesis checks are archived separately.
"""
import sys,subprocess,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/sections/figures/source_data/baseline_clinical_20261101'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 table=ROOT/'docs/paper/sections/Table_baseline_compare.tex'
 request=OUT/'finalization_request.json'
 if not request.exists():request.write_text(json.dumps(dict(manuscript_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),target=str(table),wait_pid=None),indent=2))
 phases=[('inference','evaluate_baseline_clinical_intervals.py','queue.log'),('table_refill','finalize_baseline_clinical_table.py','finalize.log')]
 for name,script,log in phases:
  (OUT/'supervisor.json').write_text(json.dumps(dict(stage=name,script=script,training=False),indent=2))
  with (OUT/log).open('a',buffering=1) as f:
   f.write('\n=== Full1600 synthesis queue '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' ===\n')
   child=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/figures'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   (OUT/('queue.pid' if name=='inference' else 'finalize.pid')).write_text(str(child.pid)+'\n')
   result=subprocess.CompletedProcess(child.args,child.wait())
  if result.returncode:
   (OUT/'SUPERVISOR_FAILED.json').write_text(json.dumps(dict(stage=name,exit_code=result.returncode,log=str(OUT/log)),indent=2));raise SystemExit(result.returncode)
 (OUT/'supervisor.json').write_text(json.dumps(dict(stage='COMPLETE',training=False,manifest=str(OUT/'FINALIZATION_COMPLETE.json')),indent=2));print('Frozen inference and verified table refill complete',flush=True)
if __name__=='__main__':main()
