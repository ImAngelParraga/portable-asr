# Private Companion Repository

Use a separate private repository for host-specific operations. Do not add it as a submodule, subtree, vendored directory, or remote that public contributors need.

Suggested name:

```text
asr-private-ops
```

## Purpose

The private companion preserves deployment knowledge that is useful to one environment but unsafe or irrelevant in a public repository.

## Suggested Layout

```text
asr-private-ops/
├── README.md
├── AGENTS.md
├── env/
│   ├── production.asr.env
│   └── overlays/
├── deploy/
│   ├── apply-public-update.sh
│   ├── install-production.sh
│   └── warmup.sh
├── docs/
│   ├── hardware.md
│   ├── network.md
│   ├── latency.md
│   ├── incidents.md
│   └── rollback.md
└── snapshots/
    └── README.md
```

## Ownership Boundary

Public repository owns:

- portable code
- sanitized config templates
- generic docs
- tests
- reusable install scripts

Private companion owns:

- live env values
- private hostnames and network exposure
- GPU UUIDs and hardware layout
- deployment commands for specific machines
- operational incidents and latency baselines

## Sync Workflow

1. Pull public repository updates.
2. Review public release notes and changed config examples.
3. Compare private env overlays against new `config/*.env.example`.
4. Apply private overlays on target host.
5. Run private warmup and latency checks.
6. Record any host-specific fix in private companion first.
7. Port only generic fixes back to public repo.

## Leak Prevention

- Never commit private companion files into public repository.
- Never make public history contain private values and then "delete" them later.
- Keep private repo outside public repo tree unless path is ignored.
- Do not add private repo as a submodule; submodule URLs and paths can reveal private structure.
- Use placeholders in public docs: `example-host`, `GPU-xxxxxxxx`, `ASR_BEARER_TOKEN=change-me`.

## Public Backport Rule

If a private issue applies to other users, translate it into generic code, config, or docs before opening a public change. Remove hostnames, GPU UUIDs, local paths, tokens, and one-machine assumptions.
