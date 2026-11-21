# -*- coding: utf-8 -*-
"""在 Trae 内运行：为离线看板导出六种真实配置；任一核验失败即退出。"""
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import warnings
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/practice/dashboard_export"
WEB = ROOT / "student_webapp"
FC = ROOT / "outputs/practice/feature_compare"
ST = ROOT / "outputs/practice/stability"
BASE = ROOT / "outputs/practice/baseline"
DATA = ROOT / "data/mall_customers_yuan.csv"
SEEDS = (0, 42, 2026)
FEATURES = {"two_features": ["income", "balance"],
            "three_features": ["income", "balance", "age"]}
RAW = ["customer_id", "age", "income", "balance", "product_count", "gender", "city"]
EXPECTED_SHA = "060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("禁止将 NaN/Infinity 导出为 JSON")
    if isinstance(value, Path):
        return str(value)
    return value


def overlap(a, b):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    assert len(a) == len(b) == 300
    table = np.zeros((3, 3), dtype=int)
    np.add.at(table, (a, b), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    matched = int(table[rows, cols].sum())
    return {"ari": float(adjusted_rand_score(a, b)), "matched": matched,
            "changed": 300 - matched, "total": 300,
            "mapping_a_to_b": mapping, "table": table.tolist()}


def parse_choose_k():
    """只解析既有日志，不重算；保留日志数值的原精度。"""
    text = (ROOT / "reports/05_choose_k.log").read_text(encoding="utf-8-sig")
    sse_part = text.split("【肘部法】", 1)[1].split("肘部法拐点", 1)[0]
    sil_part = text.split("【轮廓系数】", 1)[1].split("轮廓系数最高点", 1)[0]
    pattern = r"^\s*(\d+)\s+(\d+\.\d+)(?:\s|$)"
    sse = {int(k): float(v) for k, v in re.findall(pattern, sse_part, re.M)}
    sil = {int(k): float(v) for k, v in re.findall(pattern, sil_part, re.M)}
    if set(sse) != set(range(1, 10)) or set(sil) != set(range(2, 10)):
        raise ValueError("选k日志解析不完整，停止，不补造数值")
    elbow = int(re.search(r"肘部法拐点[^\n]*k\s*=\s*(\d+)", text).group(1))
    logged_best = int(re.search(r"轮廓系数最高点[^\n]*k\s*=\s*(\d+)", text).group(1))
    best = max(sil, key=sil.get)
    if best != logged_best:
        raise ValueError("选k日志最大值与说明不一致")
    return {"source": "reports/05_choose_k.log", "features": FEATURES["two_features"],
            "seed": 42, "standardized": True, "sse_decimal_places": 2,
            "silhouette_decimal_places": 4, "elbow_k": elbow,
            "silhouette_best_k": best,
            "elbow_method": "原脚本对归一化曲线使用首尾连线最大距离启发式",
            "rows": [{"k": k, "inertia": sse[k], "silhouette": sil.get(k)}
                     for k in range(1, 10)],
            "note": "肘点与轮廓最高值不一致。按已确认三簇路线分析，不声称k=3在所有指标上最优，也不判断差异显著性。"}


def main():
    checks = []

    def check(condition, name, details=None):
        if not bool(condition):
            raise RuntimeError("核验失败：" + name + "；" + str(details))
        checks.append({"name": name, "passed": True, "details": plain(details)})
        print("[通过]", name, "" if details is None else str(details))

    source_files = [
        DATA, BASE / "baseline_labels.csv", BASE / "baseline_metrics.json",
        FC / "feature_labels.csv", FC / "metrics.json", FC / "contingency_table.csv",
        ST / "metrics.json", ROOT / "reports/05_choose_k.log",
        *[ST / f"labels_seed_{s}.csv" for s in SEEDS],
        ROOT / "webapp/index.html", ROOT / "evidence/08_tool_sketch.jpg"]
    assets = {
        "choose_k.png": ROOT / "outputs/05_choose_k.png",
        "feature_compare.png": FC / "feature_compare.png",
        "stability.png": ST / "stability_scatter.png"}
    source_files.extend(assets.values())
    check(all(p.is_file() for p in source_files), "依赖文件全部存在")
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    check(before[str(DATA.relative_to(ROOT))] == EXPECTED_SHA, "yuan SHA256正确")
    targets = [WEB / "results.js", OUT / "results.json", OUT / "metrics.csv",
               OUT / "labels_all.csv", OUT / "validation.json",
               *[WEB / "assets" / n for n in assets]]
    check(not any(p.exists() for p in targets), "目标文件未存在，不覆盖")
    df = pd.read_csv(DATA).sort_values("customer_id").reset_index(drop=True)
    check(list(df.columns) == RAW, "原始七列及顺序正确")
    check(len(df) == 300 and df.customer_id.is_unique, "300个唯一客户")
    check(not df.isna().any().any(), "原数据无缺失")

    fc, st, base = (read_json(FC / "metrics.json"), read_json(ST / "metrics.json"),
                    read_json(BASE / "baseline_metrics.json"))
    km_params, sc_params = fc["kmeans_params"], fc["standard_scaler_params"]
    fixed = {"n_clusters": 3, "init": "k-means++", "n_init": 10,
             "max_iter": 300, "tol": 1e-4, "algorithm": "lloyd",
             "copy_x": True, "verbose": 0}
    check(all(km_params[k] == v for k, v in fixed.items()), "KMeans固定参数正确")
    check(sc_params == {"with_mean": True, "with_std": True, "copy": True},
          "StandardScaler参数正确")
    check(base["kmeans_params"] == km_params, "基础模型参数与对照一致")
    check(fc["two_features"]["feature_order"] == FEATURES["two_features"]
          and fc["three_features"]["feature_order"] == FEATURES["three_features"],
          "两种方案特征顺序正确")
    check(all(st["kmeans_params_fixed"][k] == v for k, v in fixed.items())
          and st["standard_scaler_params"] == sc_params, "稳定性参数一致")
    check(fc["dependencies"]["yuan_sha256"] == EXPECTED_SHA
          and st["dependencies"]["yuan_sha256"] == EXPECTED_SHA,
          "既有记录数据哈希一致")

    models, labels, configs, stability = {}, {}, {}, {}
    for scheme, features in FEATURES.items():
        matrix = StandardScaler(**sc_params).fit_transform(df[features].to_numpy(dtype=float))
        models[scheme], labels[scheme] = {}, {}
        for seed in SEEDS:
            params = {**km_params, "random_state": seed}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                km = KMeans(**params)
                lab = km.fit_predict(matrix)
                silhouette = float(silhouette_score(matrix, lab))
            labels[scheme][seed] = lab
            models[scheme][seed] = {
                "inertia": float(km.inertia_), "silhouette": silhouette,
                "n_iter": int(km.n_iter_), "cluster_counts": np.bincount(lab, minlength=3).tolist(),
                "warnings": [f"{w.category.__name__}: {w.message}" for w in caught]}
            print("实际拟合", scheme, seed, models[scheme][seed])
        stability[scheme] = []
        for a, b in combinations(SEEDS, 2):
            pair = {"seed_a": a, "seed_b": b, **overlap(labels[scheme][a], labels[scheme][b])}
            stability[scheme].append(pair)
            if scheme == "three_features":
                old = next(p for p in st["pairwise"] if p["seed_a"] == a and p["seed_b"] == b)
                check(np.isclose(pair["ari"], old["ari"], atol=1e-9, rtol=1e-9)
                      and pair["matched"] == old["matched"] and pair["changed"] == old["changed"],
                      f"{scheme} seed {a}/{b}与既有稳定性一致")
        for seed in SEEDS:
            lab = labels[scheme][seed]
            dm = overlap(lab, labels[scheme][42])["mapping_a_to_b"]
            rows = []
            for c in range(3):
                group = df.loc[lab == c]
                rows.append({
                    "cluster": c, "display_cluster": dm[c], "n": len(group),
                    "share_pct": len(group) / len(df) * 100,
                    "age_mean": float(group.age.mean()),
                    "income_wan_mean": float(group.income.mean()),
                    "balance_yuan_mean": float(group.balance.mean()),
                    "balance_wan_mean": float(group.balance.mean()) / 10000,
                    "product_count_mean": float(group.product_count.mean())})
            rank = {p["cluster"]: i for i, p in enumerate(sorted(rows, key=lambda p: p["income_wan_mean"]))}
            hypotheses = [
                "可调研基础账户、费用说明等服务需求；是否存在该需求需访谈验证。",
                "可调研现有产品使用与服务缺口；是否采用差异化方案需试点验证。",
                "可调研预约式或一对一服务需求；收入和余额较高不等于已知风险偏好。"]
            for p in rows:
                p["business_name"] = (
                    f"簇 {p['cluster']}｜收入{p['income_wan_mean']:.2f}万、余额"
                    f"{p['balance_wan_mean']:.2f}万（描述性命名，待本人核对）")
                p["suggestion"] = hypotheses[rank[p["cluster"]]]
            customers = df.copy()
            customers["cluster"] = lab
            customers["display_cluster"] = [dm[int(c)] for c in lab]
            key = f"{scheme}_{seed}"
            configs[key] = {
                "id": key, "scheme": scheme, "seed": seed, "features": features,
                "k": 3, "standardized": True, "kmeans_params": {**km_params, "random_state": seed},
                "scaler_params": sc_params, "metrics": models[scheme][seed],
                "display_mapping_to_seed42": dm, "profiles": rows,
                "customers": customers.to_dict(orient="records"),
                "projection": "三维聚类的二维投影" if len(features) == 3 else "二维聚类结果",
                "csv_columns": RAW + ["cluster", "display_cluster", "feature_scheme", "features",
                                      "random_state", "k", "standardized", "n_init"]}
            check(sum(p["n"] for p in rows) == 300 and len(customers) == 300,
                  f"{key}人数和客户覆盖正确")

    def verify_reference(scheme, seed, path, column, metrics):
        reference = pd.read_csv(path).sort_values("customer_id").reset_index(drop=True)
        check(len(reference) == 300 and reference.customer_id.is_unique
              and np.array_equal(reference.customer_id.to_numpy(), df.customer_id.to_numpy()),
              f"{scheme}/{seed}参照客户ID一致")
        same = overlap(labels[scheme][seed], reference[column].to_numpy())
        check(same["ari"] >= 1 - 1e-12 and same["changed"] == 0,
              f"{scheme}/{seed}逐客户分组一致", same)
        actual = models[scheme][seed]
        for metric in ("inertia", "silhouette"):
            check(np.isclose(actual[metric], metrics[metric], atol=1e-9, rtol=1e-9),
                  f"{scheme}/{seed} {metric}一致")
        check(actual["n_iter"] == metrics["n_iter"], f"{scheme}/{seed} n_iter一致")
        checks.append({"name": f"{scheme}/{seed}原始编号直接比较已记录（不作分组一致的必要条件）",
                       "passed": True, "details": bool(np.array_equal(
                           labels[scheme][seed], reference[column].to_numpy()))})

    verify_reference("two_features", 42, BASE / "baseline_labels.csv", "cluster",
                     base["metrics_from_original_run"])
    for seed in SEEDS:
        verify_reference("three_features", seed, ST / f"labels_seed_{seed}.csv",
                         f"cluster_{seed}", st["per_seed"][str(seed)])
    verify_reference("three_features", 42, FC / "feature_labels.csv", "cluster_3f",
                     fc["three_features"])

    age = overlap(labels["two_features"][42], labels["three_features"][42])
    old_age = fc["comparison"]
    check(np.isclose(age["ari"], old_age["ari"], atol=1e-9, rtol=1e-9)
          and age["matched"] == old_age["max_overlap_matched"]
          and age["changed"] == old_age["max_overlap_changed"],
          "年龄对照ARI和匹配/变化人数一致", age)
    old_table = pd.read_csv(FC / "contingency_table.csv")
    table = np.zeros((3, 3), dtype=int)
    for row in old_table.itertuples(index=False):
        table[int(row.cluster_two_features), int(row.cluster_three_features)] = int(row.n)
    check(np.array_equal(table, np.asarray(age["table"])), "年龄对照原始交叉表一致")
    choose_k = parse_choose_k()
    check(len(choose_k["rows"]) == 9, "已从原日志解析完整选k指标")
    check({str(p.relative_to(ROOT)): sha(p) for p in source_files} == before,
          "计算及核验期间所有源文件哈希未变")

    payload = {
        "schema_version": 1,
        "meta": {
            "exported_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "source": "教师第九课课堂演示包，300位教学模拟客户",
            "data_sha256": EXPECTED_SHA, "source_hashes": before, "n_customers": 300,
            "default_config": "two_features_42", "raw_columns": RAW,
            "feature_schemes": FEATURES, "random_states": list(SEEDS),
            "parameters": fixed, "scaler_params": sc_params,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                         "scipy": scipy.__version__},
            "notes": ["预计算实验结果切换，网页不重新训练模型。",
                      "income原始单位为万元，balance原始单位为元；图中余额换算为万元。",
                      "颜色仅在同一特征方案内对齐到seed42；跨方案同色不能直接比较成员。",
                      "三维模型显示收入—余额二维投影，投影重叠不能单独判定分群错误。",
                      "不同维度inertia不直接比较优劣；三个种子一致不证明全局最优。",
                      "客群业务名称和建议为待本人核对、待验证的描述与服务假设。"]},
        "configs": configs, "age_comparison": age, "stability": stability,
        "choose_k": choose_k,
        "assets": {name: f"assets/{name}" for name in assets}}
    payload = plain(payload)
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    check(len(payload["configs"]) == 6, "六个真实配置已生成")
    validation = {"checks": checks, "source_hashes": before,
                  "passed": True, "manual_acceptance": "待本人确认"}
    OUT.mkdir(parents=True, exist_ok=True)
    (WEB / "assets").mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(serialized + "\n", encoding="utf-8")
    (WEB / "results.js").write_text("window.EXPERIMENT_RESULTS = " + serialized + ";\n", encoding="utf-8")
    metric_rows, label_frames = [], []
    for cfg in payload["configs"].values():
        metric_rows.append({"config_id": cfg["id"], "scheme": cfg["scheme"], "seed": cfg["seed"],
                            "features": "|".join(cfg["features"]),
                            "cluster_counts": json.dumps(cfg["metrics"]["cluster_counts"]),
                            "inertia": cfg["metrics"]["inertia"],
                            "silhouette": cfg["metrics"]["silhouette"],
                            "n_iter": cfg["metrics"]["n_iter"],
                            "warnings": " | ".join(cfg["metrics"]["warnings"])})
        frame = pd.DataFrame(cfg["customers"])
        frame["config_id"] = cfg["id"]
        frame["features"] = "|".join(cfg["features"])
        frame["random_state"] = cfg["seed"]
        label_frames.append(frame)
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(label_frames, ignore_index=True).to_csv(OUT / "labels_all.csv", index=False, encoding="utf-8-sig")
    for name, source in assets.items():
        shutil.copy2(source, WEB / "assets" / name)
    (OUT / "validation.json").write_text(
        json.dumps(plain(validation), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8")
    print("导出完成：6配置，标签共1800行，每配置300客户。人工验收待本人确认。")
    print("结果：", WEB / "results.js")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
