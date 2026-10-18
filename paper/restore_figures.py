"""Extract historic figures from the separately tracked thesis-figures archive.

Intentionally refuses path traversal and never downloads untrusted content.
"""
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / 'figures_original_thesis.zip'
OUTPUT = ROOT / 'figures'
NEEDED = {'page33_image1.jpeg', 'page33_image2.jpeg', 'page64_image1.png', 'page67_image1.png'}

def main():
    if not ZIP.exists():
        raise SystemExit('Missing figures_original_thesis.zip; download from research Drive.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ZIP) as src:
        available = {Path(x.filename).name: x for x in src.infolist() if not x.is_dir()}
        if not NEEDED.issubset(available):
            raise SystemExit(f'Missing images: {sorted(NEEDED - set(available))}')
        for name in sorted(NEEDED):
            (OUTPUT / name).write_bytes(src.read(available[name]))
            print('extracted', name)

if __name__ == '__main__':
    main()