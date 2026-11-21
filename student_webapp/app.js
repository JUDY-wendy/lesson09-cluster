"use strict";
(() => {
  const D = window.EXPERIMENT_RESULTS;
  const $ = id => document.getElementById(id);
  const COLORS = ["#cd8151", "#49799d", "#548d72"];
  const F = (v, digits = 2) => v == null ? "—" : Number(v).toFixed(digits);
  const schemeName = s => s === "two_features" ? "收入 + 余额" : "收入 + 余额 + 年龄";
  const el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text != null) n.textContent = String(text);
    if (cls) n.className = cls;
    return n;
  };
  const sv = (tag, attrs = {}, text) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text != null) n.textContent = String(text);
    return n;
  };
  const clear = n => n.replaceChildren();
  let activePanel = "age";
  function current() {
    const key = $("featureSelect").value + "_" + $("seedSelect").value;
    if (!D || !D.configs || !D.configs[key]) throw new Error("当前配置没有对应的真实结果");
    return D.configs[key];
  }
  function table(headers, rows, caption) {
    const wrap = el("div", null, "table-wrap"), t = el("table");
    if (caption) t.append(el("caption", caption));
    const head = el("thead"), tr = el("tr");
    headers.forEach(h => tr.append(el("th", h))); head.append(tr); t.append(head);
    const body = el("tbody");
    rows.forEach(row => {const r = el("tr"); row.forEach(v => r.append(el("td", v))); body.append(r);});
    t.append(body); wrap.append(t); return wrap;
  }
  function renderChart(cfg) {
    const svg = $("scatterPlot"); clear(svg);
    const W = 760, H = 450, M = {left:65,right:20,top:18,bottom:62};
    const raw = cfg.customers;
    const x0 = Math.min(0, Math.floor(Math.min(...raw.map(c=>c.income))/10)*10);
    const x1 = Math.ceil(Math.max(...raw.map(c=>c.income))/10)*10;
    const y0 = Math.floor(Math.min(...raw.map(c=>c.balance/10000))/10)*10;
    const y1 = Math.ceil(Math.max(...raw.map(c=>c.balance/10000))/10)*10;
    const sx = x => M.left+(x-x0)/(x1-x0)*(W-M.left-M.right);
    const sy = y => H-M.bottom-(y-y0)/(y1-y0)*(H-M.top-M.bottom);
    for(let x=x0; x<=x1; x+=10) {
      svg.append(sv("line",{x1:sx(x),x2:sx(x),y1:M.top,y2:H-M.bottom,stroke:"#e7eded"}));
      svg.append(sv("text",{x:sx(x),y:H-M.bottom+23,"text-anchor":"middle",fill:"#64747a","font-size":12},x));
    }
    for(let y=y0; y<=y1; y+=10) {
      svg.append(sv("line",{x1:M.left,x2:W-M.right,y1:sy(y),y2:sy(y),stroke:"#e7eded"}));
      svg.append(sv("text",{x:M.left-12,y:sy(y)+4,"text-anchor":"end",fill:"#64747a","font-size":12},y));
    }
    svg.append(sv("line",{x1:M.left,x2:W-M.right,y1:H-M.bottom,y2:H-M.bottom,stroke:"#819599"}));
    svg.append(sv("line",{x1:M.left,x2:M.left,y1:M.top,y2:H-M.bottom,stroke:"#819599"}));
    for (const c of raw) {
      const circle=sv("circle",{cx:sx(c.income),cy:sy(c.balance/10000),r:4.1,
        fill:COLORS[c.display_cluster],opacity:.78,stroke:"#fff","stroke-width":.6,
        class:"customer","data-customer-id":c.customer_id,"data-cluster":c.cluster});
      circle.append(sv("title",{}, "客户 "+c.customer_id+"｜原簇 "+c.cluster+
        "｜年龄 "+c.age+" 岁｜收入 "+F(c.income)+" 万元｜余额 "+F(c.balance/10000)+" 万元"));
      svg.append(circle);
    }
    for(const p of cfg.profiles) {
      const cx=sx(p.income_wan_mean),cy=sy(p.balance_wan_mean),points=[];
      for(let i=0;i<10;i++) {const angle=-Math.PI/2+i*Math.PI/5,r=i%2===0?10:4.5;
        points.push((cx+Math.cos(angle)*r)+","+(cy+Math.sin(angle)*r));}
      const star=sv("polygon",{points:points.join(" "),fill:COLORS[p.display_cluster],
        stroke:"#17333b","stroke-width":1.3,class:"centroid"});
      star.append(sv("title",{},"簇 "+p.cluster+" 均值：收入 "+F(p.income_wan_mean)+
        " 万元，余额 "+F(p.balance_wan_mean)+" 万元")); svg.append(star);
    }
    svg.append(sv("text",{x:(M.left+W-M.right)/2,y:H-12,"text-anchor":"middle",fill:"#314e56","font-size":14},"年收入 income（万元）"));
    svg.append(sv("text",{x:17,y:(M.top+H-M.bottom)/2,transform:"rotate(-90 17 "+((M.top+H-M.bottom)/2)+")",
      "text-anchor":"middle",fill:"#314e56","font-size":14},"账户平均余额 balance（万元）"));
    const legend=$("chartLegend"); clear(legend);
    for(const p of [...cfg.profiles].sort((a,b)=>a.display_cluster-b.display_cluster)) {
      const item=el("span",null,"legend-item"), dot=el("span",null,"swatch");
      dot.style.backgroundColor=COLORS[p.display_cluster]; item.append(dot,document.createTextNode("簇 "+p.cluster+" · "+p.n+" 人")); legend.append(item);
    }
    svg.setAttribute("aria-label",schemeName(cfg.scheme)+"，seed "+cfg.seed+" 的300位客户收入余额散点图");
  }
  function renderProfiles(cfg) {
    const box=$("profileCards"), adv=$("businessSuggestions"); clear(box); clear(adv);
    for(const p of [...cfg.profiles].sort((a,b)=>a.income_wan_mean-b.income_wan_mean)) {
      const card=el("article",null,"profile"), top=el("div",null,"profile-top");
      card.dataset.cluster=String(p.cluster);
      const dot=el("span",null,"swatch"); dot.style.backgroundColor=COLORS[p.display_cluster];
      top.append(dot,el("h3",p.business_name)); card.append(top);
      const stats=el("div",null,"profile-stats");
      const fields=[["人数 / 占比",p.n+" 人 · "+F(p.share_pct,1)+"%"],
        ["平均年龄",F(p.age_mean,1)+" 岁"],["平均收入",F(p.income_wan_mean)+" 万元"],
        ["平均余额",F(p.balance_wan_mean)+" 万元"],["平均产品数",F(p.product_count_mean)+" 个"]];
      for(const [label,value] of fields){const cell=el("div");cell.append(el("span",label,"stat-label"),el("span",value,"stat-value"));stats.append(cell);}
      card.append(stats);box.append(card);
      const advice=el("article",null,"suggestion");advice.append(el("strong","簇 "+p.cluster+" · "+p.n+" 人"),el("p",p.suggestion));adv.append(advice);
    }
  }
  function originalFigure(panel,path,title) {
    const det=el("details"), sum=el("summary",title);
    const img=el("img"); img.src=path; img.alt=title; img.className="original-figure";
    img.loading="lazy"; det.append(sum,img);panel.append(det);
  }
  function renderEvidence(cfg) {
    const panel=$("evidencePanel");clear(panel);
    panel.setAttribute("aria-labelledby","tab-"+activePanel);
    if(activePanel==="age") {
      const age=D.age_comparison, sum=el("div",null,"evidence-summary");
      for(const [value,label] of [[age.changed+"/"+age.total,"最大重叠匹配后改变分组的人数"],[F(age.ari,4),"两特征与三特征的 ARI"]]){
        const item=el("div");item.append(el("strong",value),el("span",label));sum.append(item);}
      panel.append(sum,el("p","对照固定 seed=42，仅增加年龄特征。簇号在两种模型间不具有固定业务含义，不能逐号直接比较。","muted"));
      const rows=[];
      for(const scheme of ["two_features","three_features"]) for(const p of D.configs[scheme+"_42"].profiles)
        rows.push([scheme==="two_features"?"两特征":"三特征",p.cluster,p.n,F(p.age_mean,1),F(p.income_wan_mean),F(p.balance_wan_mean),F(p.product_count_mean)]);
      panel.append(table(["方案","原簇号","人数","均龄（岁）","收入（万元）","余额（万元）","产品（个）"],rows));
      panel.append(el("p","图表仅描述模拟数据；不同维度的 inertia 不能直接比较优劣。加入年龄带来了成员变化，不等于已证明业务效果更好。","source-note"));
      originalFigure(panel,D.assets["feature_compare.png"],"查看已有的特征对照原图");
    } else if(activePanel==="stability") {
      const pairs=D.stability[cfg.scheme];
      panel.append(el("h3",schemeName(cfg.scheme)+" · 三随机种子对照"));
      panel.append(el("p","固定其余参数，仅改变 random_state；先用最大重叠匹配对齐簇编号，再比较客户分组。","muted"));
      panel.append(table(["种子比较","ARI","匹配人数","变化人数"],pairs.map(p=>[p.seed_a+" / "+p.seed_b,F(p.ari,6),p.matched,p.changed])));
      panel.append(el("p",pairs.every(p=>p.changed===0)?
        "本次所测三个种子在当前方案下分组一致；结论仅限本数据与本参数，不能推广到所有初始化，也不证明全局最优。":
        "所测种子出现分组差异，变化人数与 ARI 如实记录；不通过改名或改色消除差异。","source-note"));
      if(cfg.scheme==="three_features") originalFigure(panel,D.assets["stability.png"],"查看已有的三特征稳定性原图");
    } else {
      const k=D.choose_k;
      panel.append(el("h3","肘部法与轮廓系数给出了不同依据"));
      panel.append(el("p","肘点启发式：k="+k.elbow_k+"；轮廓系数最高：k="+k.silhouette_best_k+
        "。本课按已确认的 k=3 路线继续分析，不声称 k=3 在所有指标上最优。","muted"));
      panel.append(table(["k","SSE / inertia","轮廓系数"],k.rows.map(r=>[r.k,F(r.inertia,2),F(r.silhouette,4)])));
      panel.append(el("p","来源："+k.source+"。两特征、seed42、标准化；SSE保留日志的2位小数，轮廓系数保留4位。k=1不计算轮廓系数；未做显著性检验。肘点计算："+k.elbow_method+"。","source-note"));
      originalFigure(panel,D.assets["choose_k.png"],"查看已有的选 k 曲线原图");
    }
  }
  function csvText(cfg) {
    const quote=v=>'"'+String(v??"").replaceAll('"','""')+'"';
    const headers=cfg.csv_columns, rows=[headers.map(quote).join(",")];
    for(const c of cfg.customers) {
      const row={...c,feature_scheme:cfg.scheme,features:cfg.features.join("|"),
        random_state:cfg.seed,k:cfg.k,standardized:cfg.standardized,n_init:cfg.kmeans_params.n_init};
      rows.push(headers.map(h=>quote(row[h])).join(","));
    }
    return "\uFEFF"+rows.join("\r\n")+"\r\n";
  }
  function download() {
    const cfg=current(), blob=new Blob([csvText(cfg)],{type:"text/csv;charset=utf-8"});
    const url=URL.createObjectURL(blob), a=el("a");
    a.href=url;a.download=cfg.id+"_k"+cfg.k+"_labels.csv";a.hidden=true;
    document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);
  }
  function render() {
    const cfg=current();
    $("metricCount").textContent=String(cfg.customers.length);
    $("metricK").textContent=String(cfg.k);
    $("metricSilhouette").textContent=F(cfg.metrics.silhouette,4);
    $("metricSeed").textContent=String(cfg.seed);
    $("metricFeatures").textContent=cfg.features.length+" 个建模特征 · 已标准化";
    $("projectionLabel").textContent=cfg.projection;
    $("configSummary").textContent="当前："+schemeName(cfg.scheme)+" / seed "+cfg.seed+
      " / k="+cfg.k+" / n_init="+cfg.kmeans_params.n_init+"；仅切换已计算结果，不实时训练。";
    $("downloadDescription").textContent=schemeName(cfg.scheme)+" · seed "+cfg.seed+
      " · "+cfg.customers.length+" 位客户。余额原始列以元保存。";
    $("downloadButton").dataset.configId=cfg.id;
    document.body.dataset.configId=cfg.id;
    renderChart(cfg);renderProfiles(cfg);renderEvidence(cfg);
    window.dashboardState={configId:cfg.id,customerCount:cfg.customers.length,
      features:cfg.features.slice(),seed:cfg.seed,silhouette:cfg.metrics.silhouette,activePanel};
  }
  try {
    if(!D||Object.keys(D.configs||{}).length!==6) throw new Error("未找到六组预计算实验结果，请保持 results.js 与页面同目录。");
    $("featureSelect").value="two_features";$("seedSelect").value="42";
    $("featureSelect").addEventListener("change",render);$("seedSelect").addEventListener("change",render);
    $("resetButton").addEventListener("click",()=>{$("featureSelect").value="two_features";$("seedSelect").value="42";activePanel="age";for(const tab of document.querySelectorAll("[data-panel]"))tab.setAttribute("aria-selected",String(tab.dataset.panel==="age"));render();});
    $("downloadButton").addEventListener("click",download);
    for(const tab of document.querySelectorAll("[data-panel]")) tab.addEventListener("click",()=>{
      activePanel=tab.dataset.panel;for(const other of document.querySelectorAll("[data-panel]"))
        other.setAttribute("aria-selected",String(other===tab));renderEvidence(current());
      if(window.dashboardState)window.dashboardState.activePanel=activePanel;
    });
    const prov=$("provenanceContent");
    prov.append(el("p",D.meta.source+"。实际数据哈希 SHA256："+D.meta.data_sha256,"source-note"));
    prov.append(el("p","固定 k=3、k-means++、n_init=10、max_iter=300、tol=0.0001、Lloyd算法；每个特征方案各自 StandardScaler。默认两特征、seed42。"));
    const list=el("ul");D.meta.notes.forEach(note=>list.append(el("li",note)));prov.append(list);
    prov.append(el("p","六组结果已与既有实验逐客户核对。网页与业务解释的本人验收仍待确认；自动检查不代替本人操作。"));
    render();
    document.body.dataset.ready="true";
  } catch(error) {
    $("pageError").hidden=false;$("pageError").textContent="加载失败："+error.message;
    document.body.dataset.ready="false";console.error(error);
  }
})();
