#!/usr/bin/env python3
"""Download, validate, and audit the locked OSA GEO data inventory.

This script is designed for Python 3.10+ on a Linux or Windows compute server
and uses only the Python standard library. Files are downloaded to ``.part`` paths, validated in
full, and atomically promoted only after they pass. Existing invalid files and
unsafe partial downloads are renamed rather than deleted.

Examples
--------
Complete the missing raw/non-normalized/platform inventory::

    python osa_geo_server_pipeline.py --root /data/OSA --group completion

Force a clean re-download and full validation of GSE214865::

    python osa_geo_server_pipeline.py --root /data/OSA \
        --group gse214865 --force

Validate an existing server-side inventory without downloading::

    python osa_geo_server_pipeline.py --root /data/OSA \
        --group all --validate-only

Select individual artifacts::

    python osa_geo_server_pipeline.py --root /data/OSA \
        --select gse214865_raw gse133601_raw gpl6244
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import re
import shutil
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


USER_AGENT = "OSA-GEO-data-audit/1.0 (public NCBI GEO research data)"
MiB = 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    key: str
    dataset: str
    relative_path: str
    url: str
    kind: str
    groups: tuple[str, ...]
    expected_size: int | None = None
    expected_members: int | None = None
    platform_accession: str | None = None
    resumable: bool = True


CATALOG: tuple[Artifact, ...] = (
    Artifact(
        "gse214865_raw", "GSE214865", "geo/GSE214865_RAW.tar",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE214nnn/GSE214865/suppl/GSE214865_RAW.tar",
        "scrna_tar", ("completion", "gse214865"), 971_284_480,
    ),
    Artifact(
        "gse75097_nonnorm", "GSE75097", "geo/GSE75097_non-normalized.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75097/suppl/GSE75097_non-normalized.txt.gz",
        "nonnorm_gzip", ("completion",), 14_421_098,
    ),
    Artifact(
        "gse71356_nonnorm", "GSE71356", "geo/GSE71356_non-normalized_data.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71356/suppl/GSE71356_non-normalized_data.txt.gz",
        "nonnorm_gzip", ("completion",), 1_588_953,
    ),
    Artifact(
        "gse133601_raw", "GSE133601", "geo/GSE133601_RAW.tar",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133601/suppl/GSE133601_RAW.tar",
        "cel_tar", ("completion", "bulk_raw"), 123_648_000, 30,
    ),
    Artifact(
        "gse49800_raw", "GSE49800", "geo/GSE49800_RAW.tar",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE49nnn/GSE49800/suppl/GSE49800_RAW.tar",
        "cel_tar", ("completion", "bulk_raw"), 157_859_840, 36,
    ),
    Artifact(
        "gse135917_raw", "GSE135917", "geo/GSE135917_RAW.tar",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135917/suppl/GSE135917_RAW.tar",
        "cel_tar", ("completion", "bulk_raw"), 281_221_120, 66,
    ),
    Artifact(
        "gpl10558", "GSE75097", "annotations/GPL10558.annot.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz",
        "annotation_gzip", ("completion", "annotations"), 7_290_886,
        platform_accession="GPL10558",
    ),
    Artifact(
        "gpl2700", "GSE71356", "annotations/GPL2700.annot.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL2nnn/GPL2700/annot/GPL2700.annot.gz",
        "annotation_gzip", ("completion", "annotations"), 3_492_715,
        platform_accession="GPL2700",
    ),
    Artifact(
        "gpl6244", "GSE49800;GSE135917", "annotations/GPL6244.annot.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6244/annot/GPL6244.annot.gz",
        "annotation_gzip", ("completion", "annotations"), 7_193_437,
        platform_accession="GPL6244",
    ),
    Artifact(
        "gpl10904", "GSE75097", "annotations/GPL10904.platform.soft.txt",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10904&targ=self&view=full&form=text",
        "platform_soft", ("completion", "annotations"),
        platform_accession="GPL10904", resumable=False,
    ),
    Artifact(
        "gpl17244", "GSE133601", "annotations/GPL17244.platform.soft.txt",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL17244&targ=self&view=full&form=text",
        "platform_soft", ("completion", "annotations"),
        platform_accession="GPL17244", resumable=False,
    ),
    Artifact(
        "gse75097_matrix", "GSE75097", "geo/GSE75097_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75097/matrix/GSE75097_series_matrix.txt.gz",
        "series_matrix", ("matrices",), 3_675_729,
    ),
    Artifact(
        "gse71356_matrix", "GSE71356", "geo/GSE71356_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71356/matrix/GSE71356_series_matrix.txt.gz",
        "series_matrix", ("matrices",), 827_761,
    ),
    Artifact(
        "gse133601_matrix", "GSE133601", "geo/GSE133601_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133601/matrix/GSE133601_series_matrix.txt.gz",
        "series_matrix", ("matrices",), 3_328_039,
    ),
    Artifact(
        "gse49800_matrix", "GSE49800", "geo/GSE49800_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE49nnn/GSE49800/matrix/GSE49800_series_matrix.txt.gz",
        "series_matrix", ("matrices",), 6_967_795,
    ),
    Artifact(
        "gse135917_matrix", "GSE135917", "geo/GSE135917_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135917/matrix/GSE135917_series_matrix.txt.gz",
        "series_matrix", ("matrices",), 8_684_678,
    ),
    Artifact(
        "gse38792_raw", "GSE38792", "geo/GSE38792_RAW.tar",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE38nnn/GSE38792/suppl/GSE38792_RAW.tar",
        "cel_tar", ("independent_validation",), expected_members=18,
    ),
    Artifact(
        "gse38792_matrix", "GSE38792", "geo/GSE38792_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE38nnn/GSE38792/matrix/GSE38792_series_matrix.txt.gz",
        "series_matrix", ("independent_validation",),
    ),
    Artifact(
        "gse293433_mrna", "GSE293433", "geo/GSE293433_processed_data_mRNA.xlsx",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293433/suppl/GSE293433_processed_data_mRNA.xlsx",
        "xlsx", ("independent_validation",),
    ),
    Artifact(
        "gse293433_matrix", "GSE293433", "geo/GSE293433_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293433/matrix/GSE293433_series_matrix.txt.gz",
        "series_matrix_metadata", ("independent_validation",), expected_members=10,
    ),
)


class ValidationError(RuntimeError):
    pass


class RangeProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_target(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    target = (root_resolved / relative_path).resolve()
    if os.path.commonpath([str(root_resolved), str(target)]) != str(root_resolved):
        raise ValueError(f"Path escapes root: {relative_path}")
    return target


def quarantine(path: Path, reason: str) -> Path:
    reason = re.sub(r"[^A-Za-z0-9_-]+", "-", reason).strip("-") or "invalid"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.quarantine-{reason}-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.quarantine-{reason}-{stamp}-{counter}")
        counter += 1
    os.replace(path, candidate)
    print(f"  QUARANTINE {path.name} -> {candidate.name}", flush=True)
    return candidate


def part_meta_path(part: Path) -> Path:
    return part.with_name(part.name + ".meta.json")


def quarantine_part(part: Path, reason: str) -> Path:
    quarantined = quarantine(part, reason)
    meta = part_meta_path(part)
    if meta.exists():
        meta_target = quarantined.with_name(quarantined.name + ".meta.json")
        os.replace(meta, meta_target)
    return quarantined


def load_part_meta(part: Path) -> dict[str, Any] | None:
    meta = part_meta_path(part)
    if not meta.exists():
        return None
    try:
        value = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_part_meta(part: Path, payload: dict[str, Any]) -> None:
    meta = part_meta_path(part)
    temporary = meta.with_name(meta.name + f".tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, meta)


@contextmanager
def exclusive_root_lock(root: Path) -> Iterable[None]:
    lock_path = root / ".osa_geo_server_pipeline.lock"
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at": utc_now(),
    }
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        existing = ""
        try:
            existing = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
        raise RuntimeError(
            f"Another pipeline process may be using this root. Lock: {lock_path}; "
            f"contents: {existing or '<unreadable>'}. Remove only after confirming no job is active."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock_path.unlink()
        except (OSError, json.JSONDecodeError):
            pass


def check_free_space(
    root: Path,
    artifacts: list[Artifact],
    validate_only: bool,
    force: bool,
) -> dict[str, int]:
    free = shutil.disk_usage(root).free
    required_payload = 0 if validate_only else sum(
        artifact.expected_size or 100 * MiB
        for artifact in artifacts
        if force or not safe_target(root, artifact.relative_path).exists()
    )
    reserve = 0 if validate_only else max(1024 * MiB, required_payload // 10)
    required = required_payload + reserve
    if free < required:
        raise RuntimeError(
            f"Insufficient free space under {root}: free={free:,}, required={required:,} "
            f"(payload={required_payload:,} + reserve={reserve:,})"
        )
    return {"free_bytes": free, "required_bytes": required, "reserve_bytes": reserve}


def check_artifact_download_space(path: Path, artifact: Artifact) -> None:
    payload = artifact.expected_size or 100 * MiB
    reserve = max(512 * MiB, payload // 20)
    free = shutil.disk_usage(path.parent).free
    if free < payload + reserve:
        raise RuntimeError(
            f"Insufficient space before downloading {artifact.key}: free={free:,}, "
            f"required={payload + reserve:,}"
        )


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_parent_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip())
    if not match:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def parse_unsatisfied_content_range(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+\*/(\d+)", value.strip())
    return int(match.group(1)) if match else None


def download_once(
    artifact: Artifact,
    part: Path,
    timeout: int,
    chunk_size: int,
) -> dict[str, Any]:
    offset = part.stat().st_size if part.exists() else 0
    if artifact.expected_size is not None:
        if offset > artifact.expected_size:
            quarantine_part(part, "oversize")
            offset = 0
        elif offset == artifact.expected_size:
            return {"resumed_from": offset, "downloaded_bytes": 0}
    elif offset and not artifact.resumable:
        quarantine_part(part, "nonresumable")
        offset = 0

    saved_meta = load_part_meta(part) if offset else None
    if offset:
        metadata_matches = (
            saved_meta is not None
            and saved_meta.get("url") == artifact.url
            and saved_meta.get("expected_size") == artifact.expected_size
        )
        validator = (saved_meta or {}).get("etag") or (saved_meta or {}).get("last_modified")
        if not metadata_matches or not validator:
            quarantine_part(part, "missing-or-mismatched-remote-metadata")
            offset = 0
            saved_meta = None

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        headers["If-Range"] = str((saved_meta or {}).get("etag") or (saved_meta or {}).get("last_modified"))
    request = urllib.request.Request(artifact.url, headers=headers, method="GET")

    started = time.monotonic()
    last_report = started
    downloaded = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        content_range = parse_content_range(response.headers.get("Content-Range"))
        content_length_raw = response.headers.get("Content-Length")
        content_length = int(content_length_raw) if content_length_raw and content_length_raw.isdigit() else None
        response_etag = response.headers.get("ETag")
        response_modified = response.headers.get("Last-Modified")

        if offset:
            if status != 206 or content_range is None or content_range[0] != offset:
                raise RangeProtocolError(
                    f"Server did not honor Range from {offset}: status={status}, "
                    f"Content-Range={response.headers.get('Content-Range')!r}"
                )
            if artifact.expected_size is not None and content_range[2] not in (None, artifact.expected_size):
                raise RangeProtocolError(
                    f"Remote total changed: expected={artifact.expected_size}, remote={content_range[2]}"
                )
            if content_range[1] < content_range[0]:
                raise RangeProtocolError(f"Invalid Content-Range end: {content_range}")
            if content_range[2] is not None and content_range[1] != content_range[2] - 1:
                raise RangeProtocolError(
                    f"Range does not reach remote EOF: Content-Range={content_range}"
                )
            range_length = content_range[1] - content_range[0] + 1
            if content_length is not None and content_length != range_length:
                raise RangeProtocolError(
                    f"Content-Length/Content-Range disagree: length={content_length}, range={content_range}"
                )
            if saved_meta and saved_meta.get("etag") and response_etag and saved_meta["etag"] != response_etag:
                raise RangeProtocolError("ETag changed during resume")
            if (
                saved_meta and saved_meta.get("last_modified") and response_modified
                and saved_meta["last_modified"] != response_modified
            ):
                raise RangeProtocolError("Last-Modified changed during resume")
        elif status not in (200, 206):
            raise RuntimeError(f"Unexpected HTTP status {status}")
        elif status == 206:
            if content_range is None or content_range[0] != 0:
                raise RangeProtocolError(
                    f"Unexpected partial response for fresh download: {response.headers.get('Content-Range')!r}"
                )
            if artifact.expected_size is not None and content_range[2] not in (None, artifact.expected_size):
                raise RangeProtocolError(
                    f"Remote total changed: expected={artifact.expected_size}, remote={content_range[2]}"
                )

        expected_transfer = None
        if artifact.expected_size is not None:
            expected_transfer = artifact.expected_size - offset
            if content_length is not None and content_length != expected_transfer:
                raise RangeProtocolError(
                    f"Content-Length mismatch: expected transfer={expected_transfer}, remote={content_length}"
                )

        meta_payload = {
            "url": artifact.url,
            "expected_size": artifact.expected_size,
            "etag": response_etag or (saved_meta or {}).get("etag"),
            "last_modified": response_modified or (saved_meta or {}).get("last_modified"),
            "remote_total": content_range[2] if content_range else content_length,
            "updated_at": utc_now(),
        }
        save_part_meta(part, meta_payload)

        mode = "ab" if offset else "wb"
        with part.open(mode) as output:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_report >= 5:
                    current = offset + downloaded
                    if artifact.expected_size:
                        pct = 100.0 * current / artifact.expected_size
                        progress = f"{current:,}/{artifact.expected_size:,} bytes ({pct:.1f}%)"
                    else:
                        progress = f"{current:,} bytes"
                    speed = downloaded / max(now - started, 0.001) / MiB
                    print(f"  PROGRESS {artifact.key}: {progress}, {speed:.2f} MiB/s", flush=True)
                    last_report = now
            output.flush()
            os.fsync(output.fileno())

    actual = part.stat().st_size
    if artifact.expected_size is not None and actual != artifact.expected_size:
        raise IOError(f"Incomplete transfer: expected={artifact.expected_size}, actual={actual}")
    return {
        "http_status": status,
        "resumed_from": offset,
        "downloaded_bytes": downloaded,
        "etag": response_etag,
        "last_modified": response_modified,
        "content_range": content_range,
    }


def download_with_retries(
    artifact: Artifact,
    part: Path,
    retries: int,
    timeout: int,
    chunk_size: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        retry_after: float | None = None
        try:
            return download_once(artifact, part, timeout, chunk_size)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 416:
                remote_total = parse_unsatisfied_content_range(exc.headers.get("Content-Range"))
                local_size = part.stat().st_size if part.exists() else 0
                if (
                    local_size > 0 and remote_total == local_size
                    and artifact.expected_size in (None, remote_total)
                ):
                    return {
                        "http_status": 416,
                        "resumed_from": local_size,
                        "downloaded_bytes": 0,
                        "content_range": exc.headers.get("Content-Range"),
                    }
                if part.exists():
                    quarantine_part(part, "http-416-range")
                print(
                    f"  RANGE_416 attempt={attempt}: local={local_size}, remote={remote_total}",
                    file=sys.stderr, flush=True,
                )
            elif exc.code == 429 or 500 <= exc.code <= 599:
                retry_header = exc.headers.get("Retry-After")
                if retry_header and retry_header.isdigit():
                    retry_after = float(retry_header)
                print(
                    f"  HTTP_RETRY attempt={attempt}: HTTP {exc.code}",
                    file=sys.stderr, flush=True,
                )
            else:
                raise RuntimeError(f"Non-retryable HTTP error {exc.code}: {exc.reason}") from exc
        except RangeProtocolError as exc:
            last_error = exc
            print(f"  RANGE_REJECT attempt={attempt}: {exc}", file=sys.stderr, flush=True)
            if part.exists():
                quarantine_part(part, "range-protocol")
        except (OSError, EOFError, urllib.error.URLError) as exc:
            last_error = exc
            print(f"  RETRYABLE attempt={attempt}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if attempt <= retries:
            delay = retry_after if retry_after is not None else min(5 * (2 ** (attempt - 1)), 60)
            time.sleep(min(delay + random.uniform(0, 1), 60))
    raise RuntimeError(f"Download failed after {retries + 1} attempts: {last_error}")


def consume_binary(handle: BinaryIO, chunk_size: int) -> int:
    total = 0
    while True:
        block = handle.read(chunk_size)
        if not block:
            return total
        total += len(block)


def validated_regular_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise ValidationError("Tar archive contains no members")
    seen_casefolded: set[str] = set()
    for member in members:
        if not member.isfile():
            raise ValidationError(
                f"Tar contains a non-regular member ({member.type!r}): {member.name}"
            )
        name = member.name
        pure = PurePosixPath(name)
        if (
            not name or "\\" in name or pure.is_absolute() or ".." in pure.parts
            or re.match(r"^[A-Za-z]:", name)
        ):
            raise ValidationError(f"Unsafe tar member path: {name!r}")
        folded = name.casefold()
        if folded in seen_casefolded:
            raise ValidationError(f"Case-folding path collision in tar: {name!r}")
        seen_casefolded.add(folded)
    return members


def validate_tar_zero_tail(path: Path, members: list[tarfile.TarInfo], chunk_size: int) -> dict[str, int]:
    last_member_end = max(
        ((member.offset_data + member.size + 511) // 512) * 512 for member in members
    )
    trailing_bytes = path.stat().st_size - last_member_end
    if trailing_bytes < 1024:
        raise ValidationError(
            f"Tar lacks the required two zero end blocks: trailing_bytes={trailing_bytes}"
        )
    nonzero = 0
    with path.open("rb") as handle:
        handle.seek(last_member_end)
        remaining = trailing_bytes
        while remaining:
            block = handle.read(min(chunk_size, remaining))
            if not block:
                break
            nonzero += sum(byte != 0 for byte in block)
            remaining -= len(block)
    if remaining != 0 or nonzero != 0:
        raise ValidationError(
            f"Tar tail is truncated or nonzero: unread={remaining}, nonzero_bytes={nonzero}"
        )
    return {
        "last_member_padded_end": last_member_end,
        "zero_tail_bytes": trailing_bytes,
    }


def gzip_summary(path: Path, chunk_size: int, capture_limit: int = 2 * MiB) -> dict[str, Any]:
    uncompressed = 0
    line_count = 0
    captured = bytearray()
    with gzip.open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            uncompressed += len(block)
            line_count += block.count(b"\n")
            if len(captured) < capture_limit:
                captured.extend(block[: capture_limit - len(captured)])
    first_line = bytes(captured).splitlines()[0].decode("utf-8", errors="replace") if captured else ""
    return {
        "gzip_eof": True,
        "uncompressed_bytes": uncompressed,
        "line_count": line_count,
        "first_line": first_line[:300],
        "captured_text": bytes(captured).decode("utf-8", errors="replace"),
    }


def count_gzip_lines_from_tar(archive: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    raw = archive.extractfile(member)
    if raw is None:
        raise ValidationError(f"Cannot extract {member.name}")
    count = 0
    with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
        for _ in handle:
            count += 1
    return count


def validate_matrix_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> tuple[int, int, int]:
    raw = archive.extractfile(member)
    if raw is None:
        raise ValidationError(f"Cannot extract {member.name}")
    dimensions: tuple[int, int, int] | None = None
    observed_entries = 0
    with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
        banner = handle.readline().rstrip(b"\r\n")
        if banner != b"%%MatrixMarket matrix coordinate integer general":
            raise ValidationError(
                f"Unexpected MatrixMarket banner in {member.name}: {banner[:120]!r}"
            )
        for line in handle:
            if line.startswith(b"%") or not line.strip():
                continue
            parts = line.split()
            if dimensions is None:
                if len(parts) != 3:
                    raise ValidationError(f"Invalid MatrixMarket dimensions in {member.name}")
                dimensions = tuple(map(int, parts))
                if any(value <= 0 for value in dimensions):
                    raise ValidationError(
                        f"Non-positive MatrixMarket dimensions in {member.name}: {dimensions}"
                    )
            else:
                observed_entries += 1
    if dimensions is None:
        raise ValidationError(f"Missing MatrixMarket dimensions in {member.name}")
    if observed_entries != dimensions[2]:
        raise ValidationError(
            f"nnz mismatch in {member.name}: header={dimensions[2]}, observed={observed_entries}"
        )
    return dimensions


def validate_scrna_tar(path: Path, chunk_size: int) -> dict[str, Any]:
    with tarfile.open(path, "r:") as archive:
        members = validated_regular_tar_members(archive)
        by_sample: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
        labels: dict[str, str] = {}
        gsm_ids: set[str] = set()
        for member in members:
            name = Path(member.name).name
            match = re.fullmatch(
                r"(GSM\d+)_(.+)_(barcodes|features|matrix)\.(?:tsv|mtx)\.gz", name
            )
            if not match:
                raise ValidationError(f"Unexpected GSE214865 member: {member.name}")
            sample_id, label, kind = match.groups()
            key = f"{sample_id}_{label}"
            by_sample[key][kind] = member
            labels[key] = label
            gsm_ids.add(sample_id)

        if len(members) != 66 or len(by_sample) != 22 or len(gsm_ids) != 22:
            raise ValidationError(
                f"Expected 66 files/22 samples/22 GSMs, found {len(members)} files/"
                f"{len(by_sample)} samples/{len(gsm_ids)} GSMs"
            )

        sample_checks = []
        for sample, parts in sorted(by_sample.items()):
            if set(parts) != {"barcodes", "features", "matrix"}:
                raise ValidationError(f"Incomplete triplet for {sample}: {sorted(parts)}")
            feature_lines = count_gzip_lines_from_tar(archive, parts["features"])
            barcode_lines = count_gzip_lines_from_tar(archive, parts["barcodes"])
            n_features, n_barcodes, nnz = validate_matrix_member(archive, parts["matrix"])
            if (n_features, n_barcodes) != (feature_lines, barcode_lines):
                raise ValidationError(
                    f"Dimension mismatch for {sample}: matrix={n_features}x{n_barcodes}, "
                    f"features={feature_lines}, barcodes={barcode_lines}"
                )
            sample_checks.append({
                "sample": sample,
                "features": feature_lines,
                "barcodes": barcode_lines,
                "nnz": nnz,
            })

    tail = validate_tar_zero_tail(path, members, chunk_size)

    group_counts = Counter(
        "control" if label.lower().startswith("control")
        else "osa" if label.lower().startswith("osa") else "other"
        for label in labels.values()
    )
    if group_counts != Counter({"control": 11, "osa": 11}):
        raise ValidationError(f"Expected 11 control + 11 OSA, found {dict(group_counts)}")
    return {
        "tar_eof": True,
        "member_count": len(members),
        "sample_count": len(by_sample),
        "unique_gsm_count": len(gsm_ids),
        "group_counts": dict(group_counts),
        "sample_checks": sample_checks,
        **tail,
    }


def validate_cel_tar(path: Path, expected_members: int, chunk_size: int) -> dict[str, Any]:
    with tarfile.open(path, "r:") as archive:
        members = validated_regular_tar_members(archive)
        if len(members) != expected_members:
            raise ValidationError(
                f"Expected {expected_members} CEL files, found {len(members)}"
            )
        uncompressed_bytes = 0
        gsm_ids: set[str] = set()
        signatures: Counter[str] = Counter()
        for member in members:
            if not member.name.lower().endswith(".cel.gz"):
                raise ValidationError(f"Unexpected non-CEL member: {member.name}")
            gsm_match = re.match(r"^(GSM\d+)_.*\.CEL\.gz$", PurePosixPath(member.name).name, re.IGNORECASE)
            if not gsm_match:
                raise ValidationError(f"CEL filename lacks a GSM accession: {member.name}")
            gsm_ids.add(gsm_match.group(1).upper())
            raw = archive.extractfile(member)
            if raw is None:
                raise ValidationError(f"Cannot extract {member.name}")
            with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
                header = handle.read(4096)
                if header.startswith(b"[CEL]"):
                    signature = "text_CEL"
                elif len(header) >= 4 and int.from_bytes(header[:4], "little") == 64:
                    signature = "XDA_binary_CEL"
                elif header.startswith(b"\x3b\x01") and b"affymetrix-calvin-" in header.lower():
                    signature = "Calvin_CEL"
                else:
                    raise ValidationError(
                        f"Unrecognized CEL content signature in {member.name}: {header[:32].hex()}"
                    )
                size = len(header) + consume_binary(handle, chunk_size)
                if size <= len(header):
                    raise ValidationError(f"Empty/truncated CEL content in {member.name}")
                uncompressed_bytes += size
                signatures[signature] += 1
        if len(gsm_ids) != expected_members:
            raise ValidationError(
                f"Expected {expected_members} unique GSM accessions, found {len(gsm_ids)}"
            )
    tail = validate_tar_zero_tail(path, members, chunk_size)
    return {
        "tar_eof": True,
        "member_count": len(members),
        "unique_gsm_count": len(gsm_ids),
        "inner_gzip_ok": len(members),
        "uncompressed_cel_bytes": uncompressed_bytes,
        "cel_signatures": dict(signatures),
        **tail,
    }


def validate_series_matrix(
    path: Path,
    allow_empty_table: bool = False,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    table_begin = False
    table_end = False
    sample_columns = 0
    expression_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        in_table = False
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                table_begin = True
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                table_end = True
                break
            if in_table:
                if sample_columns == 0:
                    sample_columns = max(0, len(next(csv.reader([line], delimiter="\t", quotechar='"'))) - 1)
                else:
                    expression_rows += 1
    invalid_empty = expression_rows == 0 and not allow_empty_table
    if not table_begin or not table_end or sample_columns == 0 or invalid_empty:
        raise ValidationError(
            f"Invalid series matrix markers/dimensions: begin={table_begin}, end={table_end}, "
            f"samples={sample_columns}, rows={expression_rows}"
        )
    if expected_samples is not None and sample_columns != expected_samples:
        raise ValidationError(
            f"Series matrix sample mismatch: expected={expected_samples}, actual={sample_columns}"
        )
    return {
        "gzip_eof": True,
        "sample_columns": sample_columns,
        "expression_rows": expression_rows,
        "metadata_only": expression_rows == 0,
    }


def validate_platform_soft(path: Path, accession: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if f"^PLATFORM = {accession}" not in text:
        raise ValidationError(f"Missing ^PLATFORM = {accession}")
    if "!platform_table_begin" not in text or "!platform_table_end" not in text:
        raise ValidationError("Missing platform table markers")
    table = text.split("!platform_table_begin", 1)[1].split("!platform_table_end", 1)[0]
    rows = max(0, sum(1 for line in table.splitlines() if line.strip()) - 1)
    if rows == 0:
        raise ValidationError("Platform table is empty")
    return {"platform": accession, "platform_table_rows": rows, "platform_table_ok": True}


def validate_xlsx(path: Path) -> dict[str, Any]:
    """Validate the ZIP container and minimum OOXML workbook structure."""
    if not zipfile.is_zipfile(str(path)):
        raise ValidationError("File is not a valid XLSX/ZIP container")
    with zipfile.ZipFile(str(path), "r") as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValidationError("Corrupt XLSX member: {}".format(corrupt_member))
        names = set(archive.namelist())
        required = {"[Content_Types].xml", "xl/workbook.xml"}
        missing = sorted(required - names)
        if missing:
            raise ValidationError("Missing XLSX members: {}".format(",".join(missing)))
        worksheets = sorted(
            name for name in names
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not worksheets:
            raise ValidationError("XLSX contains no worksheets")
    return {
        "xlsx_container_ok": True,
        "xlsx_members": len(names),
        "xlsx_worksheets": len(worksheets),
    }


def validate_artifact(artifact: Artifact, path: Path, chunk_size: int) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"Missing file: {path}")
    size = path.stat().st_size
    if artifact.expected_size is not None and size != artifact.expected_size:
        raise ValidationError(
            f"Size mismatch: expected={artifact.expected_size}, actual={size}"
        )

    if artifact.kind == "scrna_tar":
        details = validate_scrna_tar(path, chunk_size)
    elif artifact.kind == "cel_tar":
        if artifact.expected_members is None:
            raise AssertionError("CEL artifact lacks expected_members")
        details = validate_cel_tar(path, artifact.expected_members, chunk_size)
    elif artifact.kind == "series_matrix":
        details = validate_series_matrix(path)
    elif artifact.kind == "series_matrix_metadata":
        details = validate_series_matrix(
            path,
            allow_empty_table=True,
            expected_samples=artifact.expected_members,
        )
    elif artifact.kind in {"nonnorm_gzip", "annotation_gzip"}:
        details = gzip_summary(path, chunk_size)
        captured = details.pop("captured_text")
        if artifact.kind == "nonnorm_gzip" and not details["first_line"].startswith("ID_REF"):
            raise ValidationError("Non-normalized matrix does not start with ID_REF")
        if artifact.kind == "annotation_gzip":
            if details["first_line"].strip() != "^Annotation":
                raise ValidationError("GEO annotation does not start with ^Annotation")
            if artifact.platform_accession and artifact.platform_accession not in captured:
                raise ValidationError(
                    f"Annotation header does not contain {artifact.platform_accession}"
                )
    elif artifact.kind == "platform_soft":
        if artifact.platform_accession is None:
            raise AssertionError("Platform SOFT artifact lacks accession")
        details = validate_platform_soft(path, artifact.platform_accession)
    elif artifact.kind == "xlsx":
        details = validate_xlsx(path)
    else:
        raise AssertionError(f"Unknown validation kind: {artifact.kind}")

    return {"size_bytes": size, **details}


def process_artifact(
    artifact: Artifact,
    root: Path,
    validate_only: bool,
    force: bool,
    retries: int,
    timeout: int,
    chunk_size: int,
) -> dict[str, Any]:
    final = safe_target(root, artifact.relative_path)
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "key": artifact.key,
        "dataset": artifact.dataset,
        "relative_path": artifact.relative_path,
        "url": artifact.url,
        "kind": artifact.kind,
        "expected_size": artifact.expected_size,
        "started_at": utc_now(),
    }

    print(f"[{artifact.key}] {artifact.relative_path}", flush=True)
    try:
        if final.exists():
            if force and not validate_only:
                result["quarantined_final"] = str(quarantine(final, "force-redownload").relative_to(root))
            else:
                try:
                    validation = validate_artifact(artifact, final, chunk_size)
                    digest = sha256_file(final, chunk_size)
                    result.update({
                        "status": "pass",
                        "action": "validated_existing",
                        "validation": validation,
                        "sha256": digest,
                    })
                    print(f"  PASS existing sha256={digest}", flush=True)
                    return result
                except Exception as exc:
                    if validate_only:
                        raise
                    result["invalid_existing_error"] = f"{type(exc).__name__}: {exc}"
                    result["quarantined_final"] = str(
                        quarantine(final, "failed-validation").relative_to(root)
                    )

        if force and part.exists():
            result["quarantined_part"] = str(
                quarantine_part(part, "force-redownload").relative_to(root)
            )

        if validate_only:
            raise ValidationError("File is missing and --validate-only was requested")

        check_artifact_download_space(final, artifact)
        remote: dict[str, Any] = {}
        # One clean automatic recovery is allowed if a completed .part fails content validation.
        for content_attempt in (1, 2):
            if part.exists() and artifact.expected_size is not None and part.stat().st_size > artifact.expected_size:
                quarantine_part(part, "oversize")
            remote = download_with_retries(
                artifact, part, retries=retries, timeout=timeout, chunk_size=chunk_size
            )
            try:
                validation = validate_artifact(artifact, part, chunk_size)
                break
            except Exception as exc:
                if part.exists():
                    rejected = quarantine_part(part, "failed-content-validation")
                    result.setdefault("rejected_parts", []).append(str(rejected.relative_to(root)))
                if content_attempt == 2:
                    raise ValidationError(
                        f"Downloaded content failed validation twice: {type(exc).__name__}: {exc}"
                    ) from exc
                print(f"  CLEAN_RETRY after validation failure: {exc}", file=sys.stderr, flush=True)
        else:  # pragma: no cover
            raise AssertionError("Unreachable content validation loop")

        digest = sha256_file(part, chunk_size)
        os.replace(part, final)
        meta = part_meta_path(part)
        warnings = []
        try:
            if meta.exists():
                meta.unlink()
        except OSError as exc:
            warnings.append(f"Could not remove completed part metadata: {exc}")
        try:
            fsync_parent_directory(final)
        except OSError as exc:
            warnings.append(f"Could not fsync parent directory: {exc}")
        result.update({
            "status": "pass",
            "action": "downloaded_validated_promoted",
            "remote": remote,
            "validation": validation,
            "sha256": digest,
        })
        if warnings:
            result["warnings"] = warnings
        print(f"  PASS promoted sha256={digest}", flush=True)
    except Exception as exc:
        result.update({
            "status": "failed",
            "action": "not_promoted",
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"  FAILED {result['error']}", file=sys.stderr, flush=True)
    finally:
        result["finished_at"] = utc_now()
    return result


def select_artifacts(group: str, selected_keys: list[str]) -> list[Artifact]:
    by_key = {artifact.key: artifact for artifact in CATALOG}
    if selected_keys:
        unknown = sorted(set(selected_keys) - set(by_key))
        if unknown:
            raise ValueError(f"Unknown --select keys: {', '.join(unknown)}")
        return [by_key[key] for key in selected_keys]
    if group == "all":
        return list(CATALOG)
    return [artifact for artifact in CATALOG if group in artifact.groups]


def write_reports(root: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = audit_dir / f"osa_geo_audit_{stamp}.json"
    tsv_path = audit_dir / f"osa_geo_audit_{stamp}.tsv"
    payload = {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "root": str(root.resolve()),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "catalog": [asdict(item) for item in CATALOG],
        "summary": {
            "selected": len(results),
            "passed": sum(item.get("status") == "pass" for item in results),
            "failed": sum(item.get("status") != "pass" for item in results),
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = [
        "key", "dataset", "relative_path", "kind", "status", "action",
        "expected_size", "actual_size", "sha256", "url", "error",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = dict(item)
            row["actual_size"] = (item.get("validation") or {}).get("size_bytes", "")
            writer.writerow(row)
    return json_path, tsv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely download and fully validate the locked OSA GEO inventory."
    )
    parser.add_argument("--root", type=Path, required=True, help="OSA data root on the server")
    parser.add_argument(
        "--group", choices=["completion", "gse214865", "bulk_raw", "annotations", "matrices", "independent_validation", "all"],
        default="completion", help="Predefined artifact group (default: completion)",
    )
    parser.add_argument(
        "--select", nargs="*", default=[], metavar="KEY",
        help="Artifact keys; overrides --group",
    )
    parser.add_argument("--validate-only", action="store_true", help="Do not download or rename files")
    parser.add_argument("--force", action="store_true", help="Quarantine finals and download selected files again")
    parser.add_argument("--retries", type=int, default=5, help="Retries after the first transfer attempt")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP socket timeout in seconds")
    parser.add_argument("--chunk-mib", type=int, default=4, help="Streaming chunk size in MiB")
    parser.add_argument("--list", action="store_true", help="List catalog keys and exit")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retries < 0 or args.timeout <= 0 or args.chunk_mib <= 0:
        parser.error("--retries must be >=0; --timeout and --chunk-mib must be >0")
    if args.force and args.validate_only:
        parser.error("--force and --validate-only cannot be used together")
    if args.list:
        for artifact in CATALOG:
            print(
                f"{artifact.key}\t{artifact.dataset}\t{artifact.relative_path}\t"
                f"groups={','.join(artifact.groups)}"
            )
        return 0

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts = select_artifacts(args.group, args.select)
    if not artifacts:
        parser.error(f"No artifacts selected for group {args.group!r}")

    print(f"OSA GEO pipeline root: {root}", flush=True)
    print(f"Selected artifacts: {len(artifacts)}", flush=True)
    try:
        space = check_free_space(root, artifacts, args.validate_only, args.force)
        print(
            f"Free-space preflight: free={space['free_bytes']:,}, "
            f"required={space['required_bytes']:,}", flush=True,
        )
        with exclusive_root_lock(root):
            results = [
                process_artifact(
                    artifact,
                    root=root,
                    validate_only=args.validate_only,
                    force=args.force,
                    retries=args.retries,
                    timeout=args.timeout,
                    chunk_size=args.chunk_mib * MiB,
                )
                for artifact in artifacts
            ]
            json_path, tsv_path = write_reports(root, results, args)
    except RuntimeError as exc:
        print(f"PRECHECK/LOCK FAILED: {exc}", file=sys.stderr)
        return 2
    failed = [item["key"] for item in results if item.get("status") != "pass"]
    print(f"Audit JSON: {json_path}")
    print(f"Audit TSV:  {tsv_path}")
    if failed:
        print(f"FAILED artifacts: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("ALL SELECTED ARTIFACTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
