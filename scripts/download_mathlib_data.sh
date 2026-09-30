#!/bin/bash
# Download and pre-process Mathlib declaration data.
# Run once on any new machine (local Mac or EC2).
#
# Produces:
#   data/mathlib_declarations/declaration-data.bmp      (61MB — names + kinds)
#   data/mathlib_declarations/declarations_with_types.json (130MB — names + types + docs)
#
# Usage:
#   bash scripts/download_mathlib_data.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p data/mathlib_declarations

echo "=== Step 1: Download declaration-data.bmp (61MB) ==="
if [ -f data/mathlib_declarations/declaration-data.bmp ]; then
    echo "  Already exists, skipping."
else
    curl -# "https://leanprover-community.github.io/mathlib4_docs/declarations/declaration-data.bmp" \
        -o data/mathlib_declarations/declaration-data.bmp
    echo "  Done."
fi

echo ""
echo "=== Step 2: Download header-data.bmp (1.1GB) ==="
if [ -f /tmp/header-data-full.bmp ]; then
    echo "  Already exists at /tmp/header-data-full.bmp, skipping download."
else
    curl -# "https://leanprover-community.github.io/mathlib4_docs/declarations/header-data.bmp" \
        -o /tmp/header-data-full.bmp
    echo "  Done."
fi

echo ""
echo "=== Step 3: Pre-process into declarations_with_types.json (130MB) ==="
if [ -f data/mathlib_declarations/declarations_with_types.json ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
import json, re, time

print('  Loading header-data (1.1GB)...')
start = time.time()
with open('/tmp/header-data-full.bmp') as f:
    data = json.load(f)
print(f'  Loaded {len(data)} entries in {time.time()-start:.1f}s')

def strip_html(html):
    clean = re.sub(r'<[^>]+>', '', html)
    clean = clean.replace('\\\\n', ' ')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

print('  Processing...')
clean_data = {}
for name, entry in data.items():
    header_html = entry.get('header', '')
    info = entry.get('info', {})
    clean_data[name] = {
        'type': strip_html(header_html),
        'doc': info.get('doc', '')[:500],
        'kind': info.get('kind', ''),
    }

print(f'  Saving {len(clean_data)} entries...')
with open('data/mathlib_declarations/declarations_with_types.json', 'w') as f:
    json.dump(clean_data, f, separators=(',', ':'))
print('  Done.')
"
fi

echo ""
echo "=== Complete ==="
ls -lh data/mathlib_declarations/
