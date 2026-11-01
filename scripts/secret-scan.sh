#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
file_list=$(mktemp)
trap 'rm -f "$file_list"' EXIT
git ls-files -z > "$file_list"

# Report locations only: matching lines may contain credentials.
patterns=(
    '/home/[[:alnum:]_.-]+/'
    '/Users/[[:alnum:]_.-]+/'
    '[[:alnum:]_-]+\.ts\.net'
    'GPU-[0-9a-fA-F-]{36}'
    '(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3})'
    '^[[:space:]]*(export[[:space:]]+)?(ASR_BEARER_TOKEN|API_KEY|api_key|bearer_token)[[:space:]]*=[[:space:]]*[^[:alnum:]]*[A-Za-z0-9_./+-]{24,}'
    '(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,})'
    '-----BEGIN ([A-Z]+ )?PRIVATE KEY-----'
)

failed=0
while IFS= read -r -d '' file; do
    [ -f "$file" ] || continue
    [ "$file" = scripts/secret-scan.sh ] && continue
    for pattern in "${patterns[@]}"; do
        if matches=$(LC_ALL=C grep -aEn -- "$pattern" "$file"); then
            while IFS= read -r match; do
                printf '%s:%s: possible private data\n' "$file" "${match%%:*}"
            done <<< "$matches"
            failed=1
        fi
    done
done < "$file_list"

if [ "$failed" -ne 0 ]; then
    echo "secret scan failed" >&2
    exit 1
fi

echo "secret scan ok (tracked working-tree files; history and binary formats need separate review)"
