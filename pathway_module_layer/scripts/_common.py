#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared, dependency-free helpers for the OSA pathway-module layer.

Python 3.7 compatible.  Standard library only (no numpy / pandas / scipy),
matching the rest of the OSA_TRIAGE_Fuse project so the same server env runs it.

This module carries: deterministic IO (tsv / gzip-tsv / atomic json), SHA-256
auditing, GMT gene-set parsing, average ranking, Benjamini-Hochberg FDR, and an
O(k) pre-ranked GSEA enrichment-score kernel with a seeded permutation null.
Nothing here reads validation data; blinding is enforced by the callers.
"""

from __future__ import print_function

import csv
import gzip
import hashlib
import io
import json
import math
import os
from datetime import datetime


SHARED_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Time / hashing / paths
# --------------------------------------------------------------------------- #
def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path(path, root):
    ap = os.path.abspath(str(path))
    ar = os.path.abspath(str(root))
    try:
        return os.path.relpath(ap, ar).replace("\\", "/")
    except ValueError:
        return ap.replace("\\", "/")


# --------------------------------------------------------------------------- #
# Tabular IO (deterministic, gzip mtime pinned to 0 for stable hashes)
# --------------------------------------------------------------------------- #
def _opener(path, mode):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, encoding="utf-8", newline="")
    return open(str(path), mode, encoding="utf-8", newline="")


def read_tsv(path):
    """Return (rows, fieldnames).  Transparently handles .gz."""
    with _opener(path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def iter_tsv(path):
    """Stream rows as dicts without materialising the whole file."""
    with _opener(path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def write_tsv(path, rows, fields):
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t",
            lineterminator="\n", extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_tsv(path, rows, fields):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fields, delimiter="\t",
                    lineterminator="\n", extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(tmp, str(path))


def atomic_json(path, value):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))
        handle.write("\n")
    os.replace(tmp, str(path))


def _to_float_safe(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, ndigits=10):
    """Compact, locale-independent float formatting for reproducible tables."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return repr(round(value, ndigits))
    return str(value)


# --------------------------------------------------------------------------- #
# Gene sets (MSigDB-style GMT)
# --------------------------------------------------------------------------- #
def parse_gmt(path):
    """Parse a .gmt / .gmt.gz file into an ordered list of gene sets.

    Returns a list of dicts: {name, description, genes(list, de-duped,
    upper-cased, order-preserving)}.
    """
    sets = []
    with _opener(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            desc = parts[1].strip()
            seen = set()
            genes = []
            for token in parts[2:]:
                g = token.strip().upper()
                if g and g not in seen:
                    seen.add(g)
                    genes.append(g)
            if genes:
                sets.append({"name": name, "description": desc, "genes": genes})
    return sets


def detect_gmt_id_space(sets, sample=50):
    """Heuristically decide whether GMT gene tokens are Entrez ids or symbols."""
    numeric = 0
    total = 0
    for gs in sets[:sample]:
        for g in gs["genes"]:
            total += 1
            if g.isdigit():
                numeric += 1
    if total == 0:
        return "symbol"
    return "entrez" if (numeric / float(total)) > 0.8 else "symbol"


# --------------------------------------------------------------------------- #
# Ranking + multiple testing
# --------------------------------------------------------------------------- #
def average_ranks(values):
    """Return average (tie-corrected) ranks, smallest value -> rank 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def benjamini_hochberg(pvalues):
    """Return BH-adjusted q-values in the input order."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = pvalues[idx] * n / (rank + 1.0)
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    return q


# --------------------------------------------------------------------------- #
# Pre-ranked GSEA (weighted Kolmogorov-Smirnov, Subramanian 2005)
# --------------------------------------------------------------------------- #
def _es_from_positions(positions, weight_at_pos, total_weight, n_total, n_hits):
    """O(k) enrichment score from sorted 1-based hit positions.

    positions      : sorted ascending list of 1-based ranks of member genes
    weight_at_pos  : callable pos(1-based) -> |score|^p  (only queried at hits)
    total_weight   : sum of member weights (must be > 0)
    Returns (ES, peak_index_in_positions, running_at_peak).
    """
    n_miss = n_total - n_hits
    if n_miss <= 0 or total_weight <= 0.0:
        return 0.0, -1, 0.0
    miss_step = 1.0 / n_miss
    cum_hit = 0.0
    max_dev = 0.0
    min_dev = 0.0
    peak_j = -1
    for j, pos in enumerate(positions):
        # trough just before this hit: misses so far = pos-1-j, hits so far = j
        pre = cum_hit / total_weight - (pos - 1 - j) * miss_step
        if pre < min_dev:
            min_dev = pre
        cum_hit += weight_at_pos(pos)
        post = cum_hit / total_weight - (pos - j) * miss_step
        if post > max_dev:
            max_dev = post
            peak_j = j
    if abs(max_dev) >= abs(min_dev):
        return max_dev, peak_j, max_dev
    return min_dev, peak_j, min_dev


class RankedList(object):
    """A frozen ranked gene list for one contrast; supports fast GSEA."""

    def __init__(self, gene_ids, scores, weight_exponent=1.0):
        # sort by score descending, ties broken by gene id for determinism
        order = sorted(range(len(gene_ids)),
                       key=lambda i: (-scores[i], gene_ids[i]))
        self.genes = [gene_ids[i] for i in order]
        self.scores = [scores[i] for i in order]
        self.n = len(self.genes)
        self.pos_of_gene = {g: (i + 1) for i, g in enumerate(self.genes)}
        p = weight_exponent
        # weight table indexed by 1-based position
        self._w = [0.0] + [abs(self.scores[i]) ** p for i in range(self.n)]

    def _weight_at(self, pos):
        return self._w[pos]

    def enrichment(self, member_genes):
        """Observed ES + leading edge for a gene set (iterable of gene ids)."""
        positions = sorted(self.pos_of_gene[g] for g in member_genes
                           if g in self.pos_of_gene)
        k = len(positions)
        if k == 0:
            return None
        total_w = sum(self._w[pos] for pos in positions)
        es, peak_j, _ = _es_from_positions(
            positions, self._weight_at, total_w, self.n, k)
        # leading edge = member genes up to the peak (ES>=0) or from it (ES<0)
        leading = []
        if peak_j >= 0:
            if es >= 0:
                cutoff = positions[peak_j]
                leading = [self.genes[pos - 1] for pos in positions
                           if pos <= cutoff]
            else:
                cutoff = positions[peak_j]
                leading = [self.genes[pos - 1] for pos in positions
                           if pos >= cutoff]
        return {"es": es, "n_hits": k, "leading_edge": leading,
                "hit_positions": positions, "total_weight": total_w}

    def permutation_null(self, k, n_perm, rng):
        """Return list of null ES for random gene sets of size k."""
        null = []
        universe = self.n
        wtab = self._w
        for _ in range(n_perm):
            positions = sorted(rng.sample(range(1, universe + 1), k))
            total_w = 0.0
            for pos in positions:
                total_w += wtab[pos]
            es, _, _ = _es_from_positions(
                positions, self._weight_at, total_w, universe, k)
            null.append(es)
        return null


def normalized_es_and_p(observed_es, null_es):
    """Standard same-sign NES + empirical two-sided-style p-value."""
    pos = [x for x in null_es if x >= 0]
    neg = [x for x in null_es if x < 0]
    if observed_es >= 0:
        mean_pos = (sum(pos) / len(pos)) if pos else 0.0
        nes = observed_es / mean_pos if mean_pos > 0 else 0.0
        more = sum(1 for x in null_es if x >= observed_es)
        p = (more + 1.0) / (len(null_es) + 1.0)
    else:
        mean_neg = (sum(neg) / len(neg)) if neg else 0.0
        nes = observed_es / abs(mean_neg) if mean_neg < 0 else 0.0
        more = sum(1 for x in null_es if x <= observed_es)
        p = (more + 1.0) / (len(null_es) + 1.0)
    return nes, p
