from __future__ import annotations

import csv
import gzip
import json
import sqlite3
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def split_geo(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))


def audit_series_matrix(path: Path) -> dict:
    rows: dict[str, list[list[str]]] = defaultdict(list)
    expression_header = None
    expression_rows = 0
    first_feature = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        in_table = False
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                break
            if in_table:
                fields = split_geo(line)
                if expression_header is None:
                    expression_header = fields
                else:
                    expression_rows += 1
                    if first_feature is None and fields:
                        first_feature = fields[0]
                continue
            if line.startswith("!"):
                fields = split_geo(line)
                rows[fields[0]].append(fields[1:])

    accessions = (rows.get("!Sample_geo_accession") or [[]])[0]
    titles = (rows.get("!Sample_title") or [[]])[0]
    platforms = (rows.get("!Sample_platform_id") or [[]])[0]
    sources = (rows.get("!Sample_source_name_ch1") or [[]])[0]
    characteristics = rows.get("!Sample_characteristics_ch1", [])
    characteristic_counts: dict[str, Counter] = defaultdict(Counter)
    for characteristic_row in characteristics:
        for value in characteristic_row:
            if ":" in value:
                key, val = value.split(":", 1)
                characteristic_counts[key.strip().lower()][val.strip()] += 1

    return {
        "file": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "series_title": ((rows.get("!Series_title") or [[""]])[0] or [""])[0],
        "n_samples": len(accessions),
        "platforms": dict(Counter(platforms)),
        "sources": dict(Counter(sources)),
        "titles": titles,
        "characteristics": {k: dict(v) for k, v in characteristic_counts.items()},
        "expression_rows": expression_rows,
        "expression_columns": max(0, len(expression_header or []) - 1),
        "first_feature": first_feature,
    }


def audit_tar(path: Path) -> dict:
    with tarfile.open(path, "r:*", ignore_zeros=True) as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
    suffix_counts = Counter("".join(Path(m.name).suffixes) or "<none>" for m in members)
    return {
        "file": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "n_files": len(members),
        "suffix_counts": dict(suffix_counts),
        "first_files": [m.name for m in members[:30]],
    }


def audit_sqlite(path: Path) -> dict:
    result = {"file": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size}
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
            tables = [r[0] for r in conn.execute(
                "select name from sqlite_master where type='table' order by name"
            )]
            result["sqlite"] = True
            result["tables"] = tables
            counts = {}
            for table in tables[:100]:
                escaped = table.replace('"', '""')
                try:
                    counts[table] = conn.execute(f'select count(*) from "{escaped}"').fetchone()[0]
                except sqlite3.DatabaseError:
                    pass
            result["row_counts"] = counts
            selected = [
                "ReportInfo",
                "DataTypeSubsets",
                "DataTypeSubsetDetails",
                "TargetProteins_Subset_1",
                "QuanSpectrumInfo_WorkflowIDAsc_SpectrumIDAsc",
                "MSnSpectrumInfo_WorkflowIDAsc_SpectrumIDAsc",
            ]
            result["selected_schemas"] = {}
            result["selected_examples"] = {}
            for table in selected:
                if table not in tables:
                    continue
                escaped = table.replace('"', '""')
                result["selected_schemas"][table] = [
                    {"name": row[1], "type": row[2]}
                    for row in conn.execute(f'pragma table_info("{escaped}")')
                ]
                examples = conn.execute(f'select * from "{escaped}" limit 2').fetchall()
                result["selected_examples"][table] = [
                    [value.hex()[:80] if isinstance(value, bytes) else value for value in row]
                    for row in examples
                ]
    except sqlite3.DatabaseError as exc:
        result["sqlite"] = False
        result["error"] = str(exc)
        result["magic_hex"] = path.read_bytes()[:32].hex()
    return result


def main() -> None:
    report = {
        "series_matrices": [audit_series_matrix(p) for p in sorted((ROOT / "geo").glob("*series_matrix.txt.gz"))],
        "single_cell_archives": [audit_tar(p) for p in sorted((ROOT / "geo").glob("*_RAW.tar"))],
        "proteomics": [],
    }
    for path in sorted((ROOT / "PXD032734").glob("*")):
        if path.suffix.lower() == ".msfview":
            report["proteomics"].append(audit_sqlite(path))
        elif path.is_file():
            report["proteomics"].append({
                "file": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
