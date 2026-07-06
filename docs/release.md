# Release Checklist

Use this before making the repository public or cutting a public release.

## Public Repository

1. Confirm all host-specific files are outside this repository or ignored.
2. Run the secret scan:

   ```bash
   scripts/secret-scan.sh
   ```

3. Run tests:

   ```bash
   .venv/bin/python -m unittest discover -s tests
   ```

4. Check shell syntax for install helpers:

   ```bash
   bash -n scripts/install-service.sh
   bash -n scripts/install-llm-runtime.sh
   bash -n scripts/diagnose-postprocess.sh
   ```

5. Confirm no generated files are pending:

   ```bash
   git status --short
   ```

6. Review changed docs and config examples for placeholders only.

## Private Companion

1. Pull or fetch the public branch into a private test checkout.
2. Compare private env overlays with `config/asr.env.example` and `config/profiles/*.env.example`.
3. Update private deployment notes for any new or renamed settings.
4. Apply overlays on the target host.
5. Restart the private service using private runbook commands.
6. Warm transcription and post-processing models if that host uses lazy loading.
7. Record private latency, GPU, network, or rollback notes in the private companion repo.

## Backport Rule

If a private fix would help other users, remove private values and port it back as a generic public change. Keep real hostnames, tokens, GPU UUIDs, private paths, and incident details in the private companion only.
