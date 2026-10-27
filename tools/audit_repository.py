"""Check syntax, portable paths, text language and packaged data integrity."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import gzip
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'outputs', '.git', '.venv', '__pycache__', 'resources', 'results', 'raw', 'metadata', 'tmp'}
TEXT_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.md', '.txt', '.tsv', '.csv', '.sh', '.r', '.ps1', '.toml', '.log'}
MACHINE_PATH = re.compile(r'(?<![A-Za-z])(?:[A-Za-z]:[/\\]|/(?:srv|home|Users)/)')
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}')

def check_text(name, text, problems):
    if re.search(r'[\u3400-\u9fff]', text):
        problems.append(name + ': non-English CJK text')
    if MACHINE_PATH.search(text):
        problems.append(name + ': machine-specific absolute path')
    if SECRET.search(text):
        problems.append(name + ': possible credential')

def main():
    problems = []
    count = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p in SKIP for p in relative.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {'.gitignore', '.gitattributes'}:
            continue
        text = path.read_text(encoding='utf-8-sig')
        count += 1
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as exc:
                problems.append(str(relative) + ': syntax: ' + str(exc))
        check_text(str(relative), text, problems)
    manifest = ROOT / 'docs/source_manifest.json'
    for record in json.loads(manifest.read_text(encoding='utf-8'))['files']:
        path = ROOT / record['path']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record['portable_sha256']:
            problems.append(record['path'] + ': source checksum differs from snapshot')
    archive_files = 0
    archive_manifest = ROOT / 'source_data/manifest.json'
    if archive_manifest.is_file():
        for record in json.loads(archive_manifest.read_text(encoding='utf-8'))['archives']:
            path = ROOT / 'source_data' / record['file']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                problems.append(record['file'] + ': archive checksum mismatch')
                continue
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    problems.append(record['file'] + ': corrupt archive member')
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    archive_files += 1
                    name = record['file'] + ':' + info.filename
                    if '..' in Path(info.filename).parts or info.filename.startswith('/'):
                        problems.append(name + ': unsafe member name')
                    if Path(info.filename).suffix.lower() in TEXT_SUFFIXES:
                        check_text(name, archive.read(info).decode('utf-8-sig'), problems)
                    elif info.filename.endswith(('.tsv.gz', '.csv.gz', '.txt.gz')):
                        check_text(name, gzip.decompress(archive.read(info)).decode('utf-8-sig'), problems)
    report = {'text_files_checked': count, 'archive_members_checked': archive_files,
              'problems': problems, 'status': 'PASS' if not problems else 'FAIL'}
    print(json.dumps(report, indent=2))
    return int(bool(problems))

if __name__ == '__main__':
    raise SystemExit(main())
