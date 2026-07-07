#!/usr/bin/env bash
set -euo pipefail

patterns=(
    'javiserver'
    'tailf[[:alnum:]]*\.ts\.net'
    'GPU-[0-9a-fA-F-]{36}'
    '/home/rankis'
    '/opt/rankis'
    'ASR_BEARER_TOKEN=[A-Za-z0-9_./+-]{12,}'
    'Jellyfin'
    'hermes-stt'
    'hermes-llm'
    '192\.168\.150\.1'
)

allow_file() {
    case "$1" in
        scripts/secret-scan.sh)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

failed=0
while IFS= read -r file; do
    [ -f "$file" ] || continue
    allow_file "$file" && continue
    for pattern in "${patterns[@]}"; do
        if LC_ALL=C grep -En "$pattern" "$file" >/tmp/asr-secret-scan-match.$$; then
            cat /tmp/asr-secret-scan-match.$$
            failed=1
        fi
        rm -f /tmp/asr-secret-scan-match.$$
    done
done < <(git ls-files)

if [ "$failed" -ne 0 ]; then
    echo "secret scan failed" >&2
    exit 1
fi

echo "secret scan ok"
