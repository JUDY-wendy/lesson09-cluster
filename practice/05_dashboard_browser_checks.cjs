"use strict";
const fs=require("fs"),path=require("path"),{spawn}=require("child_process"),{pathToFileURL}=require("url");
const root=path.resolve(__dirname,".."),web=path.join(root,"student_webapp"),work=path.join(root,"outputs/practice/dashboard_browser_qa");
const reportPath=path.join(root,"reports/dashboard_browser_checks.json");
const D=JSON.parse(fs.readFileSync(path.join(root,"outputs/practice/dashboard_export/results.json"),"utf8"));
const report={started_at:new Date().toISOString(),passed:false,checks:[],screenshots:[],downloads:[],manual_acceptance:"待本人操作，自动检查不代替本人验收"};
let child,bc,pc,downloads,errors=[],external=[];
function check(name,ok,detail=""){report.checks.push({name,passed:!!ok,detail});if(!ok)throw Error(name+": "+detail);}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){const end=Date.now()+25000;while(Date.now()<end){const v=await fn();if(v)return v;await pause(100);}throw Error("等待超时："+label);}
class CDP{
 constructor(ws){this.ws=ws;this.id=0;this.pending=new Map();this.listeners=[];
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=this.pending.get(m.id);if(p){clearTimeout(p.timer);this.pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else this.listeners.forEach(f=>f(m));};}
 static async open(url){const ws=new WebSocket(url);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});return new CDP(ws);}
 send(method,params={}){return new Promise((resolve,reject)=>{const id=++this.id,timer=setTimeout(()=>{this.pending.delete(id);reject(Error("CDP timeout "+method));},15000);this.pending.set(id,{resolve,reject,timer});this.ws.send(JSON.stringify({id,method,params}));});}
 close(){this.ws.close();}
}
async function ev(expression){const r=await pc.send("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function ready(){await until(()=>ev("document.body?.dataset.ready==='true'||!!document.getElementById('pageError')?.textContent"),"网页加载");check("页面加载完成",await ev("document.body.dataset.ready==='true'"),await ev("document.getElementById('pageError')?.textContent"));}
async function select(s,seed){await ev("document.getElementById('featureSelect').value="+JSON.stringify(s)+";document.getElementById('seedSelect').value="+JSON.stringify(String(seed))+";document.getElementById('featureSelect').dispatchEvent(new Event('change',{bubbles:true}))");}
async function current(cfg){
 const actual=await ev("({id:document.body.dataset.configId,button:document.getElementById('downloadButton').dataset.configId,count:document.getElementById('metricCount').textContent,k:document.getElementById('metricK').textContent,seed:document.getElementById('metricSeed').textContent,sil:document.getElementById('metricSilhouette').textContent,points:[...document.querySelectorAll('circle.customer')].map(x=>({id:x.dataset.customerId,cluster:x.dataset.cluster,fill:x.getAttribute('fill')})),stars:document.querySelectorAll('polygon.centroid').length,profiles:[...document.querySelectorAll('.profile')].map(x=>({cluster:x.dataset.cluster,text:x.textContent}))})");
 check(cfg.id+" 配置与指标联动",actual.id===cfg.id&&actual.button===cfg.id&&actual.count==="300"&&actual.k==="3"&&actual.seed===String(cfg.seed)&&actual.sil===cfg.metrics.silhouette.toFixed(4));
 const map=new Map(actual.points.map(x=>[x.id,x])),colors=["#cd8151","#49799d","#548d72"];
 check(cfg.id+" 散点逐客户标签与配色",map.size===300&&actual.points.length===300&&actual.stars===3&&cfg.customers.every(c=>{const p=map.get(String(c.customer_id));return p&&p.cluster===String(c.cluster)&&p.fill===colors[c.display_cluster];}));
 check(cfg.id+" 画像对应真实均值",actual.profiles.length===3&&cfg.profiles.every(p=>{const t=actual.profiles.find(x=>x.cluster===String(p.cluster))?.text||"";return t.includes(p.n+" 人")&&t.includes(p.income_wan_mean.toFixed(2)+" 万元")&&t.includes(p.balance_wan_mean.toFixed(2)+" 万元")&&t.includes(p.age_mean.toFixed(1)+" 岁")&&t.includes(p.product_count_mean.toFixed(2)+" 个");}));
}
function csv(text){text=text.replace(/^\uFEFF/,"");let rows=[],row=[],field="",quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}else if(c===","&&!quoted){row.push(field);field="";}else if((c==="\r"||c==="\n")&&!quoted){if(c==="\r"&&text[i+1]==="\n")i++;row.push(field);if(row.some(x=>x!==""))rows.push(row);row=[];field="";}else field+=c;}if(field||row.length){row.push(field);rows.push(row);}return rows;}
async function download(cfg){
 const before=new Set(fs.readdirSync(downloads));await ev("document.getElementById('downloadButton').click()");
 const file=await until(()=>fs.readdirSync(downloads).find(f=>!before.has(f)&&f.endsWith(".csv")),"下载 CSV");
 const full=path.join(downloads,file),buffer=fs.readFileSync(full),rows=csv(buffer.toString("utf8")),header=rows.shift();
 check(cfg.id+" CSV 编码、表头、300行",buffer.subarray(0,3).equals(Buffer.from([239,187,191]))&&JSON.stringify(header)===JSON.stringify(cfg.csv_columns)&&rows.length===300,file);
 const actual=new Map(rows.map(r=>[r[header.indexOf("customer_id")],Object.fromEntries(header.map((h,i)=>[h,r[i]]))]));
 check(cfg.id+" CSV 每个字段与当前模型一致",actual.size===300&&cfg.customers.every(c=>{const row={...c,feature_scheme:cfg.scheme,features:cfg.features.join("|"),random_state:cfg.seed,k:cfg.k,standardized:cfg.standardized,n_init:cfg.kmeans_params.n_init},a=actual.get(String(c.customer_id));return a&&header.every(h=>a[h]===String(row[h]??""));}));
 report.downloads.push(path.relative(root,full));
}
async function screenshot(name){const m=await pc.send("Page.getLayoutMetrics"),size=m.cssContentSize||m.contentSize;const r=await pc.send("Page.captureScreenshot",{format:"png",captureBeyondViewport:true,clip:{x:0,y:0,width:size.width,height:size.height,scale:1}});const f=path.join(root,"evidence",name);if(fs.existsSync(f))throw Error("拒绝覆盖截图 "+f);fs.writeFileSync(f,Buffer.from(r.data,"base64"));report.screenshots.push(path.relative(root,f));}
async function viewport(width,height){await pc.send("Emulation.setDeviceMetricsOverride",{width,height,deviceScaleFactor:1,mobile:false});await pause(150);}
(async()=>{try{
 if(fs.existsSync(reportPath))throw Error("拒绝覆盖已有浏览器报告");
 fs.mkdirSync(work,{recursive:true});downloads=path.join(work,"downloads");fs.mkdirSync(downloads,{recursive:true});check("下载目录初始为空",fs.readdirSync(downloads).length===0);
 const profile=path.join(work,"edge-profile"),edge="C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
 child=spawn(edge,["--headless=new","--no-first-run","--no-default-browser-check","--disable-background-networking","--disable-sync","--disable-component-update","--remote-debugging-port=0","--remote-allow-origins=*","--user-data-dir="+profile,"about:blank"],{windowsHide:true,stdio:["ignore",fs.openSync(path.join(work,"edge.stdout.log"),"w"),fs.openSync(path.join(work,"edge.stderr.log"),"w")]});
 child.on("error",e=>errors.push(String(e)));
 const active=path.join(profile,"DevToolsActivePort");await until(()=>fs.existsSync(active),"启动浏览器");const [port,browserPath]=fs.readFileSync(active,"utf8").trim().split(/\r?\n/);
 bc=await CDP.open("ws://127.0.0.1:"+port+browserPath);
 const target=await (await fetch("http://127.0.0.1:"+port+"/json/new?about:blank",{method:"PUT"})).json();pc=await CDP.open(target.webSocketDebuggerUrl);
 pc.listeners.push(m=>{if(m.method==="Runtime.exceptionThrown")errors.push(JSON.stringify(m.params));if(m.method==="Runtime.consoleAPICalled"&&m.params.type==="error")errors.push(JSON.stringify(m.params));if(m.method==="Network.requestWillBeSent"&&/^https?:/i.test(m.params.request.url))external.push(m.params.request.url);});
 for(const method of ["Page.enable","Runtime.enable","Network.enable"])await pc.send(method);
 await bc.send("Browser.setDownloadBehavior",{behavior:"allow",downloadPath:downloads});
 await viewport(1440,1000);await pc.send("Page.navigate",{url:pathToFileURL(path.join(web,"index.html")).href});await ready();
 check("默认两特征 seed42",await ev("document.body.dataset.configId==='two_features_42'"));
 await screenshot("09_tool_default.png");
 for(const cfg of Object.values(D.configs)){await select(cfg.scheme,cfg.seed);await current(cfg);await download(cfg);if(cfg.id==="three_features_42")await screenshot("10_tool_changed.png");}
 await ev("document.getElementById('tab-k').click()");check("选 k 数值与结论",await ev("document.querySelectorAll('#evidencePanel tbody tr').length===9&&document.getElementById('evidencePanel').textContent.includes('0.7351')&&document.getElementById('evidencePanel').textContent.includes('k=2')&&document.getElementById('evidencePanel').textContent.includes('k=3')"));
 await ev("document.getElementById('tab-age').click()");check("年龄对照真实变化与 ARI",await ev("document.getElementById('evidencePanel').textContent.includes('17/300')&&document.getElementById('evidencePanel').textContent.includes('0.8142')"));
 for(const scheme of ["two_features","three_features"]){await select(scheme,42);await ev("document.getElementById('tab-stability').click()");const expected=scheme==="two_features"?"收入 + 余额 ·":"收入 + 余额 + 年龄 ·";check(scheme+" 稳定性面板联动",await ev("document.querySelectorAll('#evidencePanel tbody tr').length===3&&document.getElementById('evidencePanel').textContent.includes("+JSON.stringify(expected)+")"));}
 await select("three_features",2026);await ev("document.getElementById('resetButton').click()");check("恢复默认配置和标签页",await ev("document.body.dataset.configId==='two_features_42'&&window.dashboardState.activePanel==='age'&&document.getElementById('tab-age').getAttribute('aria-selected')==='true'"));
 await viewport(390,844);check("390px 窄屏无整页横向溢出",await ev("document.documentElement.scrollWidth<=innerWidth+1"));await screenshot("11_tool_mobile_auto.png");
 await viewport(1440,1000);await pc.send("Network.emulateNetworkConditions",{offline:true,latency:0,downloadThroughput:-1,uploadThroughput:-1});await pc.send("Page.reload",{ignoreCache:true});await ready();await current(D.configs.two_features_42);check("断网刷新可用",true);
 const copy=path.join(work,"portable_copy/student_webapp");if(fs.existsSync(copy))throw Error("拒绝覆盖便携副本");fs.cpSync(web,copy,{recursive:true,errorOnExist:true,force:false});
 await pc.send("Page.navigate",{url:pathToFileURL(path.join(copy,"index.html")).href});await ready();await select("three_features",42);await current(D.configs.three_features_42);
 const sizes=await ev("Promise.all(['choose_k.png','feature_compare.png','stability.png'].map(f=>new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i.naturalWidth);i.onerror=()=>reject(new Error(f));i.src='assets/'+f;})))");check("复制文件夹后离线图片均可读取",sizes.length===3&&sizes.every(n=>n>0));
 downloads=path.join(work,"downloads_portable");fs.mkdirSync(downloads);await bc.send("Browser.setDownloadBehavior",{behavior:"allow",downloadPath:downloads});await download(D.configs.three_features_42);
 check("无外部网络请求",external.length===0,external.join("\n"));check("浏览器无脚本异常",errors.length===0,errors.join("\n"));
 await ev("document.getElementById('tab-age').click()");check("数据表行无整行背景填色",await ev("[...document.querySelectorAll('#evidencePanel th,#evidencePanel td')].every(x=>getComputedStyle(x).backgroundColor==='rgba(0, 0, 0, 0)')"));
 report.passed=true;
 }catch(e){report.error=e.stack;console.error(e.stack);process.exitCode=1;}
 finally{report.finished_at=new Date().toISOString();report.console_errors=errors;report.external_requests=external;
 fs.writeFileSync(reportPath,JSON.stringify(report,null,2),"utf8");
 const md="# 看板自动浏览器检查\n\n结果："+(report.passed?"通过":"失败并停止")+"\n\n本人验收：待本人操作，自动检查不代替本人验收。\n\n|检查|结果|\n|---|---|\n"+report.checks.map(c=>"|"+c.name+"|"+(c.passed?"通过":"失败")+"|").join("\n")+"\n\n"+(report.error?"错误记录：\n\n"+report.error+"\n":"")+"截图："+report.screenshots.join("、")+"\n";
 fs.writeFileSync(path.join(root,"reports/dashboard_browser_checks.md"),md,"utf8");
 if(bc){try{await bc.send("Browser.close");}catch{}bc.close();}if(pc)pc.close();if(child&&!child.killed)child.kill();
 console.log("BROWSER_CHECK_RESULT="+(report.passed?"PASS":"FAIL")+"; checks="+report.checks.length);
 }
})();