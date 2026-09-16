import json, os, glob, datetime as dt
WATCH=os.path.join(os.path.dirname(__file__),'inbox')
STATE=os.path.join(os.path.dirname(__file__),'done.json')
LOG=os.path.join(os.path.dirname(__file__),'watch.log')
done=json.load(open(STATE,encoding='utf-8')) if os.path.exists(STATE) else {}
new=[p for p in sorted(glob.glob(os.path.join(WATCH,'*'))) if os.path.basename(p) not in done]
for p in new: done[os.path.basename(p)]={'at':dt.datetime.now().isoformat(timespec='seconds')}
json.dump(done,open(STATE,'w',encoding='utf-8'),ensure_ascii=False,indent=1)
open(LOG,'a',encoding='utf-8').write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} 처리 {len(new)}건\n")
