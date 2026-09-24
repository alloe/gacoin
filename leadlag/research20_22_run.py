#!/usr/bin/env python3
"""Three bounded read-only backtests run concurrently; preserve independent Phase19."""
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
import research_core as core
import research20_news,research21_oi,research22_cross

def main():
 core.log('R2022_START',phases=[20,21,22],no_orders=True,max_hours=2,existing_service_only=True)
 statuses={}
 with ThreadPoolExecutor(max_workers=3) as pool:
  fs={pool.submit(fn):name for name,fn in [('P20',research20_news.run),('P21',research21_oi.run),('P22',research22_cross.run)]}
  for f in as_completed(fs):
   name=fs[f]
   try:f.result();statuses[name]={'status':'completed','production_pass':False}
   except Exception as e:
    statuses[name]={'status':'failed','error':repr(e)};core.log(name+'_FATAL',error=repr(e),traceback=traceback.format_exc()[-2400:])
 core.write_report('source_manifest',core.MANIFEST)
 core.log('R2022_DONE',phases=statuses,http_requests=core.NREQ,download_bytes=core.NBYTES,source_resources=len(core.MANIFEST),production_pass=False)
if __name__=='__main__':main()
