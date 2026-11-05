#!/usr/bin/env python3
"""Evaluate geometric verification rates and optional external pair GT.

Use --compare to scan finished matching runs, or --pred with --inlier-pairs
for a single run. All counts use unique undirected non-self image pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def read_pairs(path: Path, prefix: str = "") -> list[tuple[str, str]]:
    prefix = prefix.replace("\\", "/").rstrip("/")
    pairs: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(
                f"{path}:{line_number}: expected two whitespace-separated image paths"
            )
        names = []
        for value in columns:
            value = value.replace("\\", "/")
            if prefix:
                if not value.startswith(prefix + "/"):
                    raise ValueError(
                        f"{path}:{line_number}: {value!r} does not start with {prefix + '/'!r}"
                    )
                value = value[len(prefix) + 1 :]
            names.append(value)
        pairs.append((names[0], names[1]))
    return pairs


def unique_pairs(rows):
    return {tuple(sorted((a, b))) for a, b in rows if a != b}


def score_gt(predictions, ground_truth) -> dict:
    """Score unique undirected predicted pairs against an external GT set."""
    if not predictions:
        raise ValueError("Prediction file must contain at least one pair")
    if not ground_truth:
        raise ValueError("GT must contain at least one non-self pair")
    matched = len(predictions & ground_truth)
    return {
        "protocol": "undirected_pair_precision",
        "global_num": matched,
        "num_test": len(predictions),
        "accuracy": matched / len(predictions),
        "precision": matched / len(predictions),
        "recall": matched / len(ground_truth),
        "gt_pairs": len(ground_truth),
    }


def score_inliers(predictions, inliers) -> dict:
    if not predictions:
        raise ValueError("Original pair list is empty")
    extra = inliers - predictions
    if extra:
        raise ValueError(
            f"{len(extra)} inlier pairs are absent from the original pair list; check method/Top-K"
        )
    return {
        "protocol": "geometry_verification_rate",
        "input_pairs": len(predictions),
        "inlier_pairs": len(inliers),
        "failed_pairs": len(predictions) - len(inliers),
        "verification_rate": len(inliers) / len(predictions),
    }


def database_inliers(run, predictions):
    """Read geometry counts only when a run has no exported inlier lists."""
    with sqlite3.connect(
        (run / "database.db").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        names = dict(db.execute("SELECT image_id,name FROM images"))
        ids = {name: key for key, name in names.items()}
        expected = {
            min(ids[a], ids[b]) * 2147483647 + max(ids[a], ids[b])
            for a, b in predictions
        }
        matched = {row[0] for row in db.execute("SELECT pair_id FROM matches")}
        if matched != expected:
            raise ValueError(f"{run}: matching does not cover the exact input pair set")
        found = set()
        result = {15: set(), 30: set()}
        for key, count, config in db.execute(
            "SELECT pair_id,rows,config FROM two_view_geometries"
        ):
            found.add(key)
            if config in {2, 3, 4, 5, 6, 8, 9}:
                a, b = divmod(key, 2147483647)
                pair = tuple(sorted((names[a], names[b])))
                for threshold in result:
                    if count >= threshold:
                        result[threshold].add(pair)
        if found != expected:
            raise ValueError(
                f"{run}: geometric verification is incomplete or contains other input pairs"
            )
    return result


def compare_inliers(sfm_root, output_dir, topk=30):
    results = []
    labels = {
        "bow": "BoW",
        "vlad": "VLAD",
        "netvlad": "NetVLAD",
        "vlad256": "VLAD256",
        "anyloc_dinov2_vitg14_l31_value_aerial_c32_img322": "AnyLoc",
        "cosplace_resnet50_d512_img512": "CosPlace",
    }
    for manifest_path in sorted(sfm_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("topk") != topk or not manifest.get("pair_file"):
            continue
        run = manifest_path.parent
        status = (run / "status.txt").read_text().strip()
        if status != "complete":
            timing = run / "timing.tsv"
            stages = []
            if timing.is_file():
                with timing.open() as stream:
                    stages = list(csv.DictReader(stream, delimiter="\t"))
            matching_completed = any(
                row.get("stage") == "matches_importer" and row.get("exit_code") == "0"
                for row in stages
            )
            if not matching_completed:
                print(f"Skip unfinished matching: {run}", flush=True)
                continue
        snapshot = run / "pairs.txt"
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest["pair_sha256"]:
            raise ValueError(f"{run}: input snapshot hash differs from manifest")
        rows = read_pairs(snapshot)
        predictions = unique_pairs(rows)
        original = Path(manifest["pair_file"])
        if len(rows) != len(predictions) or len(predictions) != manifest["num_pairs"]:
            raise ValueError(f"{run}: input count or uniqueness differs from manifest")
        if unique_pairs(read_pairs(original)) != predictions:
            raise ValueError(
                f"{run}: original retrieval pairs differ from the matching snapshot"
            )
        exports = list(run.glob("analysis_exports/**/inlier_pairs.txt"))
        if len(exports) > 1:
            raise ValueError(
                f"{run}: multiple inlier exports; select a single export first"
            )
        source_paths = {}
        if exports and (exports[0].parent / "verified_pairs_ge30.txt").is_file():
            folder = exports[0].parent
            sets = {
                15: unique_pairs(read_pairs(exports[0])),
                30: unique_pairs(read_pairs(folder / "verified_pairs_ge30.txt")),
            }
            if (folder / "verified_pairs_ge15.txt").exists() and sets[
                15
            ] != unique_pairs(read_pairs(folder / "verified_pairs_ge15.txt")):
                raise ValueError(f"{run}: inlier_pairs.txt is not the ge15 alias")
            source_paths = {
                15: str(exports[0]),
                30: str(folder / "verified_pairs_ge30.txt"),
            }
            summary_path = folder / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary["num_undirected_pairs"] != len(predictions):
                    raise ValueError(f"{run}: export summary input count mismatch")
                for row in summary["thresholds"]:
                    if (
                        row["min_inliers"] in sets
                        and len(sets[row["min_inliers"]])
                        != row["successful_undirected_pairs"]
                    ):
                        raise ValueError(
                            f"{run}: exported inlier count differs from summary"
                        )
        else:
            sets = database_inliers(run, predictions)
            source_paths = {threshold: str(run / "database.db") for threshold in sets}
        if not sets[30] <= sets[15]:
            raise ValueError(f"{run}: ge30 is not a subset of ge15")
        method_key = manifest["method"]
        method = (
            "Ours"
            if method_key.startswith("ours_")
            else labels.get(method_key, method_key)
        )
        for threshold, inliers in sets.items():
            record = {
                "scene": manifest["dataset"],
                "method": method,
                "backend": manifest.get("retrieval_backend") or "",
                "topk": topk,
                "min_inliers": threshold,
                **score_inliers(predictions, inliers),
                "pred": str(original),
                "inlier_source": source_paths[threshold],
                "method_key": method_key,
                "input_pair_sha256": manifest["pair_sha256"],
                "run": str(run),
                "run_status": status,
                "matching_completed": True,
            }
            results.append(record)
            print(
                f"{record['scene']} {method} ge{threshold}: {len(inliers):,}/{len(predictions):,} = {record['verification_rate']:.2%}",
                flush=True,
            )
    if not results:
        raise ValueError(f"No finished Top-{topk} matching runs found under {sfm_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(output_dir, results)
    write_inlier_report(output_dir, results)


def save_summary(output_dir, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def write_inlier_report(output_dir, results):
    topk = results[0]["topk"]
    lines = [
        f"# UAVPairs Top-{topk} 几何验证通过率（不依赖 GT）",
        "",
        f"每个方法只使用自身同一轮 Top-{topk} 的原始检索 pairs 和匹配后的 inlier pairs。A–B 与 B–A 合并，排除自身配对；失败配对保留在分母中。",
        "",
        "**通过率 = 内点数达到阈值的唯一无向配对数 / 原始唯一无向配对数。**",
        "",
        "分别报告内点数 ≥15、≥30；排除 UNDEFINED、DEGENERATE、WATERMARK 几何配置。该指标是几何匹配成功率，不是外部 GT 精确率，也不是 SfM 重建成功率。",
        "",
        "| 数据集 | 方法 | 后端 | 原始配对数 | ≥15 通过数 | ≥15 通过率 | ≥30 通过数 | ≥30 通过率 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    lookup = {
        (r["scene"], r["method"], r["backend"], r["min_inliers"]): r for r in results
    }
    scenes = sorted({r["scene"] for r in results})
    methods = sorted({(r["method"], r["backend"]) for r in results})
    compact = ["## 方法对比", ""]
    compact.append(
        "| 方法 / 后端 | "
        + " | ".join(
            f"{scene} ≥{threshold}" for scene in scenes for threshold in (15, 30)
        )
        + " |"
    )
    compact.append("|---|" + "---:|" * (len(scenes) * 2))
    for method, backend in methods:
        values = []
        for scene in scenes:
            for threshold in (15, 30):
                row = lookup.get((scene, method, backend, threshold))
                values.append(f"{row['verification_rate']:.2%}" if row else "—")
        label = method + (f" / {backend}" if backend else "")
        compact.append("| " + label + " | " + " | ".join(values) + " |")
    lines[8:8] = compact + ["", "## 配对数量明细", ""]
    groups = {}
    for r in results:
        groups.setdefault(
            (r["scene"], r["method"], r["backend"], r["topk"], r["run"]), {}
        )[r["min_inliers"]] = r
    for (scene, method, backend, topk, run), values in sorted(groups.items()):
        a, b = values[15], values[30]
        lines.append(
            f"| {scene} | {method} | {backend or '—'} | {a['input_pairs']:,} | {a['inlier_pairs']:,} | {a['verification_rate']:.2%} | {b['inlier_pairs']:,} | {b['verification_rate']:.2%} |"
        )
    lines += [
        "",
        "## 每行对应的数据来源",
        "",
        "| 数据集 | 方法 | 原始检索 pairs | 匹配运行目录 | 重建状态 |",
        "|---|---|---|---|---|",
    ]
    for (scene, method, backend, _, run), values in sorted(groups.items()):
        r = values[15]
        name = method + (f" ({backend})" if backend else "")
        lines.append(
            f"| {scene} | {name} | `{r['pred']}` | `{run}` | {r['run_status']} |"
        )
    lines += [
        "",
        "JSON 的 `inlier_source` 是各阈值所用的导出文件或数据库；`pred` 是原始检索文件，`run` 是匹配运行目录，`input_pair_sha256` 对应运行时的输入快照。",
        "",
        "只统计匹配阶段已完整完成的运行；没有完整匹配的 CosPlace、VLAD256 及其他检索后端不填 0、不纳入排名。matching_complete 表示仅匹配完成；即使 mapper 失败，只要匹配与几何验证完整，仍可计算本指标，但不能据此宣称重建成功。",
        "",
        "BoW 的原始 Top30 包含自身，删除自身后每个 query 有 29 个候选；未补取第 31 个。其他方法的候选数沿用原始输入。所有通过率均按各自实际唯一无向配对数计算，不是查询级 Precision@30。",
        "",
        "[详细结果 JSON](results.json) · [表格 CSV](results.csv) · [汇总入口](../README.md)",
        "",
        "复现（项目根目录）：",
        "",
        "```bash",
        "python tools/retrieval/evaluate_retrieval_pairs.py --compare-inliers",
        "```",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def write_summary_index(root):
    root.mkdir(parents=True, exist_ok=True)
    reports = sorted(root.glob("*/report.md"))
    lines = [
        "# UAVPairs 评测汇总",
        "",
        "统一使用每种方法自身的原始检索配对和几何验证结果；不再使用外部 GT。",
        "",
    ]
    for report in reports:
        relative = report.relative_to(root).as_posix()
        lines.append(f"- [{report.parent.name}：报告与全部结果]({relative})")
    lines += [
        "",
        "几何评测目录保留 report.md、results.json、results.csv。",
        "MD 展示方法对比、实际配对数量和运行来源；JSON/CSV 保存每条指标的原始 pairs、内点来源与输入摘要。",
        "efficiency 目录保存检索分阶段计时，按数据集/方法组织；pilot 表示样本流程验证，不是全量结果。",
        "",
        "通过率 = 达到 ≥15 或 ≥30 内点阈值的唯一无向配对数 / 原始唯一无向配对数；失败配对保留在分母。",
        "匹配完成即可统计；matching_complete 不代表完成重建，mapper 失败也不会被标记为重建成功。",
        "缺少完整匹配结果的方法不填 0，也不从其他方法或 Top-K 借用内点结果。",
        "",
        "```bash",
        "python tools/retrieval/evaluate_retrieval_pairs.py --compare",
        "```",
        "",
        "旧外部 GT、依赖 GT 的精确率结果及混合报告已清理；原始检索 pairs 与 COLMAP 匹配数据保留。",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--gt", type=Path, help="External ground-truth pair file")
    parser.add_argument(
        "--gt-prefix", default="", help="Prefix to strip from GT image paths"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    parser.add_argument(
        "--inlier-pairs",
        type=Path,
        help="Compare exported inlier pairs to --pred, without external GT",
    )
    parser.add_argument(
        "--compare",
        "--compare-inliers",
        dest="compare",
        action="store_true",
        help="Compare own-run matching success rates (no external GT)",
    )
    parser.add_argument(
        "--sfm-root", type=Path, default=Path("data/UAVPairs/sfm-results")
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Top-K of runs selected by --compare-inliers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the mode-specific report directory",
    )
    args = parser.parse_args()
    if args.compare:
        if args.pred or args.gt or args.inlier_pairs or args.output:
            parser.error("--compare cannot be combined with single-file evaluation")
        output_dir = args.output_dir or Path(
            f"results/pairs/_summaries/geometry_top{args.topk}"
        )
        compare_inliers(args.sfm_root, output_dir, args.topk)
        if args.output_dir is None:
            write_summary_index(Path("results/pairs/_summaries"))
        return
    if args.pred is None:
        parser.error("Use --compare, or provide --pred with --gt or --inlier-pairs")
    if args.gt is not None and args.inlier_pairs is not None:
        parser.error("--gt and --inlier-pairs are mutually exclusive")
    if args.gt is not None:
        predictions = unique_pairs(read_pairs(args.pred))
        ground_truth = unique_pairs(read_pairs(args.gt, args.gt_prefix))
        result = {
            "pred": str(args.pred),
            "gt": str(args.gt),
            "gt_prefix": args.gt_prefix,
            **score_gt(predictions, ground_truth),
        }
    elif args.inlier_pairs is not None:
        result = {
            "pred": str(args.pred),
            "inlier_source": str(args.inlier_pairs),
            **score_inliers(
                unique_pairs(read_pairs(args.pred)),
                unique_pairs(read_pairs(args.inlier_pairs)),
            ),
        }
    else:
        parser.error("Use --compare, or provide --pred with --gt or --inlier-pairs")
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
