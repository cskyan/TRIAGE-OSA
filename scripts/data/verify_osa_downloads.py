from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


EXPECTED_SIZES = {
    "geo/GSE214865_RAW.tar": 971_284_480,
    "geo/GSE75097_non-normalized.txt.gz": 14_421_098,
    "geo/GSE71356_non-normalized_data.txt.gz": 1_588_953,
    "geo/GSE133601_RAW.tar": 123_648_000,
    "geo/GSE49800_RAW.tar": 157_859_840,
    "geo/GSE135917_RAW.tar": 281_221_120,
    "annotations/GPL10558.annot.gz": 7_290_886,
    "annotations/GPL2700.annot.gz": 3_492_715,
    "annotations/GPL6244.annot.gz": 7_193_437,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_to_eof(path: Path) -> dict:
    uncompressed = 0
    lines = 0
    first_line = ""
    with gzip.open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            if not first_line:
                first_line = block.splitlines()[0].decode("utf-8", errors="replace")[:300]
            uncompressed += len(block)
            lines += block.count(b"\n")
    return {"gzip_eof": True, "uncompressed_bytes": uncompressed, "line_count": lines, "first_line": first_line}


def count_gzip_lines_from_tar(archive: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    raw = archive.extractfile(member)
    if raw is None:
        raise ValueError(f"Cannot extract {member.name}")
    count = 0
    with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
        for _ in handle:
            count += 1
    return count


def matrix_market_shape_from_tar(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[int, int, int]:
    raw = archive.extractfile(member)
    if raw is None:
        raise ValueError(f"Cannot extract {member.name}")
    dimensions: tuple[int, int, int] | None = None
    observed_entries = 0
    with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
        for line in handle:
            if line.startswith(b"%"):
                continue
            parts = line.split()
            if not parts:
                continue
            if dimensions is None:
                if len(parts) != 3:
                    raise ValueError(f"Invalid Matrix Market dimensions in {member.name}")
                dimensions = tuple(map(int, parts))
            else:
                observed_entries += 1
    if dimensions is None:
        raise ValueError(f"Missing Matrix Market dimensions in {member.name}")
    if observed_entries != dimensions[2]:
        raise ValueError(
            f"Matrix Market nnz mismatch in {member.name}: header={dimensions[2]}, "
            f"observed={observed_entries}"
        )
    return dimensions


def nonzero_bytes(path: Path, start: int, end: int) -> int:
    if end <= start:
        return 0
    count = 0
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                break
            count += sum(byte != 0 for byte in block)
            remaining -= len(block)
    return count


def verify_scrna_tar(path: Path, expected_size: int | None = None) -> dict:
    with tarfile.open(path, "r:") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        by_sample: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
        for member in members:
            name = Path(member.name).name
            match = re.match(r"(GSM\d+)_(.+)_(barcodes|features|matrix)\.(?:tsv|mtx)\.gz$", name)
            if not match:
                raise ValueError(f"Unexpected member: {member.name}")
            sample_id, label, kind = match.groups()
            by_sample[f"{sample_id}_{label}"][kind] = member
        if len(members) != 66 or len(by_sample) != 22:
            raise ValueError(f"Expected 66 files/22 samples, found {len(members)}/{len(by_sample)}")

        last_member_end = max(
            ((member.offset_data + member.size + 511) // 512) * 512 for member in members
        )
        sample_checks = []
        for sample, parts in sorted(by_sample.items()):
            if set(parts) != {"barcodes", "features", "matrix"}:
                raise ValueError(f"Incomplete triplet for {sample}: {sorted(parts)}")
            feature_lines = count_gzip_lines_from_tar(archive, parts["features"])
            barcode_lines = count_gzip_lines_from_tar(archive, parts["barcodes"])
            n_features, n_barcodes, nnz = matrix_market_shape_from_tar(archive, parts["matrix"])
            if (n_features, n_barcodes) != (feature_lines, barcode_lines):
                raise ValueError(
                    f"Dimension mismatch for {sample}: matrix={n_features}x{n_barcodes}, "
                    f"features={feature_lines}, barcodes={barcode_lines}"
                )
            sample_checks.append({
                "sample": sample, "features": feature_lines, "barcodes": barcode_lines, "nnz": nnz
            })
    result = {
        "tar_eof": True, "member_count": len(members), "sample_count": len(by_sample),
        "last_member_padded_end": last_member_end, "sample_checks": sample_checks,
    }
    if expected_size is not None:
        result["official_region_after_last_member_nonzero_bytes"] = nonzero_bytes(
            path, last_member_end, min(expected_size, path.stat().st_size)
        )
        result["bytes_after_official_size"] = max(path.stat().st_size - expected_size, 0)
        result["extra_region_nonzero_bytes"] = nonzero_bytes(
            path, expected_size, path.stat().st_size
        )
    return result


def verify_cel_tar(path: Path, expected_members: int) -> dict:
    with tarfile.open(path, "r:") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if len(members) != expected_members:
            raise ValueError(f"Expected {expected_members} CEL members, found {len(members)}")
        suffixes = Counter("".join(Path(m.name).suffixes).lower() for m in members)
        gzip_ok = 0
        for member in members:
            raw = archive.extractfile(member)
            if raw is None:
                raise ValueError(f"Cannot extract {member.name}")
            if member.name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
                    while handle.read(1024 * 1024):
                        pass
                gzip_ok += 1
            else:
                while raw.read(1024 * 1024):
                    pass
        return {"tar_eof": True, "member_count": len(members), "suffixes": dict(suffixes), "inner_gzip_ok": gzip_ok}


def verify_soft_text(path: Path, accession: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if f"^PLATFORM = {accession}" not in text:
        raise ValueError(f"Missing platform accession {accession}")
    if "!platform_table_begin" not in text or "!platform_table_end" not in text:
        raise ValueError(f"Missing platform table markers for {accession}")
    between = text.split("!platform_table_begin", 1)[1].split("!platform_table_end", 1)[0]
    rows = sum(1 for line in between.splitlines() if line.strip()) - 1
    return {"platform": accession, "table_rows": rows, "platform_table_ok": True}


def verify_one(path: Path) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    canonical_rel = rel.removesuffix(".part")
    canonical_name = path.name.removesuffix(".part")
    result = {"path": rel, "exists": path.exists()}
    if not path.exists():
        result.update({"status": "missing"})
        return result
    result["size_bytes"] = path.stat().st_size
    expected = EXPECTED_SIZES.get(canonical_rel)
    size_mismatch = False
    if expected is not None:
        result["expected_size"] = expected
        if path.stat().st_size != expected:
            size_mismatch = True
            result["size_match"] = False
        else:
            result["size_match"] = True
    try:
        if canonical_name == "GSE214865_RAW.tar":
            result.update(verify_scrna_tar(path, expected))
        elif canonical_name in {"GSE133601_RAW.tar", "GSE49800_RAW.tar", "GSE135917_RAW.tar"}:
            expected_members = {"GSE133601_RAW.tar": 30, "GSE49800_RAW.tar": 36, "GSE135917_RAW.tar": 66}[canonical_name]
            result.update(verify_cel_tar(path, expected_members))
        elif canonical_name.endswith(".gz"):
            result.update(read_gzip_to_eof(path))
        elif canonical_name == "GPL10904.platform.soft.txt":
            result.update(verify_soft_text(path, "GPL10904"))
        elif canonical_name == "GPL17244.platform.soft.txt":
            result.update(verify_soft_text(path, "GPL17244"))
        result["sha256"] = sha256(path)
        if size_mismatch:
            result.update({"status": "failed", "error": "size_mismatch"})
        else:
            result["status"] = "pass"
    except Exception as exc:  # report exact validation failure
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Paths relative to workspace root")
    parser.add_argument("--json-output", help="Write the complete verification report as JSON")
    args = parser.parse_args()
    default_paths = [
        "geo/GSE75097_series_matrix.txt.gz",
        "geo/GSE71356_series_matrix.txt.gz",
        "geo/GSE133601_series_matrix.txt.gz",
        "geo/GSE49800_series_matrix.txt.gz",
        "geo/GSE135917_series_matrix.txt.gz",
        "geo/GSE75097_non-normalized.txt.gz",
        "geo/GSE71356_non-normalized_data.txt.gz",
        "geo/GSE133601_RAW.tar",
        "geo/GSE49800_RAW.tar",
        "geo/GSE135917_RAW.tar",
        "geo/GSE214865_RAW.tar",
        "annotations/GPL10558.annot.gz",
        "annotations/GPL2700.annot.gz",
        "annotations/GPL6244.annot.gz",
        "annotations/GPL10904.platform.soft.txt",
        "annotations/GPL17244.platform.soft.txt",
    ]
    paths = [ROOT / p for p in (args.paths or default_paths)]
    results = [verify_one(p) for p in paths]
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_output:
        output = ROOT / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if any(r["status"] != "pass" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
