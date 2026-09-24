"""Read-only data feasibility, no outcome backtests or credential use."""
import requests,json,re,io,zipfile,hashlib,time
from html.parser import HTMLParser
class H(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.text=[]
 def handle_starttag(self,t,a):
  if t=='a':self.links += [v for k,v in a if k=='href']
 def handle_data(self,s):self.text.append(s)
def log(t,**d):print(t+' '+json.dumps(d,separators=(',',':')),flush=True)
urls=[('bls_cpi','https://www.bls.gov/bls/news-release/cpi.htm'),('bls_jobs','https://www.bls.gov/bls/news-release/empsit.htm'),('bls2024','https://www.bls.gov/schedule/2024/home.htm'),('bls2025','https://www.bls.gov/schedule/2025/home.htm'),('bls2026','https://www.bls.gov/schedule/2026/home.htm'),('bea2024','https://www.bea.gov/news/schedule/2024'),('bea2025','https://www.bea.gov/news/schedule/2025'),('bea2026','https://www.bea.gov/news/schedule'),('fed','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),('nq','https://query1.finance.yahoo.com/v8/finance/chart/NQ%3DF?interval=5m&range=1mo'),('zt','https://query1.finance.yahoo.com/v8/finance/chart/ZT%3DF?interval=5m&range=1mo'),('spot','https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01-15.zip')]
for name,url in urls:
 try:
  r=requests.get(url,timeout=(10,30),allow_redirects=False,headers={'User-Agent':'DaolMacroResearch/23-public-read-only'});log('P23_PROBE_HTTP',name=name,status=r.status_code,bytes=len(r.content),location=r.headers.get('location'),sha256=hashlib.sha256(r.content).hexdigest())
  if r.status_code!=200:continue
  if name=='spot':
   z=zipfile.ZipFile(io.BytesIO(r.content));lines=z.read(z.namelist()[0]).decode().splitlines();log('P23_PROBE_SPOT',rows=len(lines),schema=lines[0].split(',')[:1],columns=len(lines[0].split(',')))
  elif name in ('nq','zt'):
   x=r.json().get('chart',{});y=(x.get('result') or [{}])[0];log('P23_PROBE_MARKET',name=name,error=x.get('error'),bars=len(y.get('timestamp',[])),meta=y.get('meta',{}),quote_keys=list((y.get('indicators',{}).get('quote') or [{}])[0]))
  else:
   h=H();h.feed(r.text);links=[x for x in h.links if re.search(r'(cpi_|empsit_|monetary202[456]|personal-income|schedule/202)',x)];txt=' '.join(' '.join(h.text).split())
   log('P23_PROBE_CALENDAR',name=name,links=links[:105],text_excerpt=txt[txt.find('January 2024'):txt.find('January 2024')+1200] if 'January 2024' in txt else txt[:1800])
 except Exception as e:log('P23_PROBE_FAIL',name=name,error=repr(e))
 time.sleep(.2)
log('P23_PROBE_DONE',no_orders=True,no_price_outcomes=True)
