# SPEC: Public-ready ASR project

## §G Goal
G1. Make repo public-safe, installable, and hardware-portable while preserving private host knowledge in separate private companion repo.

## §C Constraints
C1. Public repo must not expose hostnames, tailnet domains, GPU UUIDs, private paths, bearer tokens, sudoers assumptions, personal workflow names, or one-host incident lore.
C2. Private operational knowledge must not be lost; move it to private companion repo with clear sync path back to public core.
C3. Public install path must work for CPU-only, single Nvidia GPU, multi-GPU Nvidia, and externally hosted postprocess LLM.
C4. Defaults must be safe for unknown users: localhost bind, generated token, conservative model sizes, no required private hardware.
C5. Public docs must explain choices and tradeoffs without assuming private host names, personal client flows, private network names, media-server names, or specific personal examples.
C6. Code must keep OpenAI-compatible endpoints stable unless explicitly versioned.
C7. Transcript cleanup must remain conservative and language-preserving, but defaults/examples must be generic.
C8. Tests must not require GPU, network, live Whisper, live llama.cpp, systemd, root, or private companion repo.
C9. Install scripts must be idempotent and configurable through env vars or flags, not hardcoded to one deployment.
C10. Public repo and private companion repo must have explicit boundaries so private files cannot be committed by accident.

## §I Interfaces
I1. Public HTTP API: `GET /health`, `POST /v1/audio/transcriptions`, `POST /v1/text/postprocess`, `POST /v1/chat/completions`, `POST /v1/llm/chat/completions`.
I2. Public config surface: `ASR_*`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, env file templates, optional profile files.
I3. Installer surface: `scripts/install-service.sh`, `scripts/install-llm-runtime.sh`, `systemd/*.example`.
I4. Runtime backends: faster-whisper CPU/CUDA, llama.cpp local CLI/server, optional external OpenAI-compatible postprocess endpoint.
I5. Private companion repo surface: host overlay docs, live deployment notes, private `AGENTS.md`, env overlays, hardware inventory, Tailscale/service commands, latency baselines.
I6. Sync surface between repos: public-safe templates in core, private overlay files ignored by core, documented copy/patch workflow.
I7. Security surface: tokens, hostnames, private network names, GPU UUIDs, logs, caches, model paths, service account names.

## §V Invariants
V1. `rg -n` secret scan over public repo finds no private hostnames, tailnet domains, GPU UUIDs, bearer tokens, live env values, private user paths, or personal deployment lore.
V2. Public README describes project generically and provides quickstart for local dev plus production install without private host assumptions.
V3. `AGENTS.md` in public repo contains contributor-safe guidance only; private operations instructions live outside public repo.
V4. Public config examples bind to `127.0.0.1` by default and never include real tokens or private addresses.
V5. Service defaults support CPU-only startup when configured with `ASR_DEVICE=cpu` and do not require Nvidia tools unless CUDA path selected.
V6. GPU selection accepts empty, ordinal, or UUID values, but docs present UUID as optional advanced config, not required setup.
V7. Postprocess LLM is optional: ASR transcription works without local Qwen/llama.cpp when `ASR_POSTPROCESS_ENABLED=0`.
V8. Local postprocess model/provider is configurable; code/docs do not hardcode Qwen as only supported path.
V9. Install scripts expose service name, install dirs, model id, context, backend, listen host/port, and user as configurable inputs.
V10. Hardware-specific watcher behavior is opt-in/configurable and documented as media-server coexistence support, not default core behavior.
V11. Public examples avoid personal words/workflows as required behavior; regression tests may keep linguistic edge cases when privacy-safe.
V12. Private companion repo has checklist for applying public updates and reapplying private overlays without editing public history.
V13. `.gitignore` blocks live env files, generated overlays, logs, pid files, caches, model downloads, virtualenvs, and private companion checkouts.
V14. Unit tests cover config/profile parsing and backend selection without loading GPU models or making network calls.
V15. Public release process includes clean working tree, passing tests, secret scan, and private-overlay audit before making repo public.
V16. GPU watcher patterns match watched process executable paths, not unrelated command arguments, so idle service daemons that mention a watched binary path do not block LLM startup.

## §T Tasks
id|status|task|cites
T1|x|classify repo content into public core, generic examples, private companion, and obsolete local migration material|V1,V3,V11,I5,I7
T2|x|create private companion repo plan: files, ownership, sync workflow, and no-submodule/no-public-history leak rules|V12,I5,I6
T3|x|move private operational detail from public `AGENTS.md` into companion repo; replace public `AGENTS.md` with generic contributor/deploy guidance|V1,V3,V12,I5
T4|x|rewrite `README.md` as public project docs: overview, quickstart, CPU install, CUDA install, optional postprocess, API examples, security notes|V2,V4,V7,V8,I1,I2
T5|x|split config examples into generic base plus documented profiles: cpu, cuda-single-gpu, cuda-multi-gpu, external-postprocess, local-llama|V4,V5,V6,V7,V8,I2,I4
T6|x|make service defaults generic: localhost bind, safe cache paths, no private IP, no one-host llama paths except install-template defaults|V4,V5,V7,I2
T7|x|parameterize `scripts/install-service.sh` and generated systemd unit for service name, install dir, env path, user/group, host/port, and hardening paths|V9,I3
T8|x|parameterize `scripts/install-llm-runtime.sh` for install dir, model id, context, backend, CUDA requirement, skip download, and smoke test text|V8,V9,I3,I4
T9|x|retire or privatize private migration script unless rewritten as generic migration example|V1,V11,I3,I5
T10|x|make llama-server process discovery use configured binary/path/pid file, not hardcoded to one install path|V1,V8,V9,I4
T11|x|make media/ffmpeg GPU watcher configurable with generic process patterns and default disabled unless explicitly enabled|V10,I4
T12|x|add config loader/profile helper if needed so examples and runtime validation stay consistent|V5,V6,V7,V8,V14,I2
T13|x|add public-safe secret scan script or test for forbidden patterns and generated/private file names|V1,V13,V15,I7
T14|x|update `.gitignore` for env overlays, private notes, logs, pids, caches, models, virtualenvs, downloaded binaries, companion checkout dirs|V13,I6,I7
T15|x|add unit tests for CPU/no-postprocess config, external postprocess config, local llama config, and watcher disabled path|V5,V7,V8,V10,V14
T16|x|replace personal README examples and private phrases with generic privacy-safe examples|V1,V2,V11,I7
T17|x|document public release checklist and private companion update checklist|V12,V15,I5,I6
T18|x|run tests, secret scan, `git status --short`, then commit and push public-prep changes before repo visibility change|V15

## §B Bugs
id|date|cause|fix
B1|2026-07-06|GPU watcher matched full command line and blocked LLM on idle media-server daemon args|V16
