# Service-Owned Transcript Semantics Spec

§G
G1: Make ASR service sole owner of transcription semantics: vocabulary biasing, cleanup, filler handling, correction, and optional translation.
G2: Return final semantic transcript from audio endpoint so thin clients only present/insert received text.
G3: Select Whisper or optional Qwen3-ASR per request through existing OpenAI-compatible `model` field.
G4: Advertise currently enabled transcription models to clients without exposing the cleanup LLM as an ASR choice.

§C
C1: Keep `POST /v1/audio/transcriptions` OpenAI-compatible for existing clients.
C2: No client-specific vocabulary, fuzzy threshold, filler filter, or post-processing model required in desktop client.
C3: Cleanup remains conservative, meaning-preserving, language-preserving unless translation explicitly requested.
C4: Post-processing remains optional; disabled/unavailable optional cleanup fails open to raw Whisper text unless request selects required behavior.
C5: Vocabulary context is bounded, sanitized, excluded from logs, and must not silently become permanent server state.
C6: Translation never occurs implicitly. Request must select translation and target semantics explicitly.
C7: Tests require no GPU, network, live Whisper, or live LLM.
C8: Existing model names continue to select Whisper; Qwen3-ASR runs only when explicitly enabled and selected.

§I
I.transcribe: `POST /v1/audio/transcriptions` accepts multipart audio, `model`, `temperature`, optional `language`, optional bounded `prompt`; returns final text after configured service processing.
I.translate: Add explicit translation API contract before enabling client controls: OpenAI-compatible `POST /v1/audio/translations` for English, with separately documented/versioned extension if arbitrary target languages are supported.
I.postprocess: `POST /v1/text/postprocess` and `POST /v1/chat/completions` expose service cleanup directly; audio endpoint may invoke same internal pipeline.
I.config: `ASR_POSTPROCESS_*` owns provider/model/token/timeout/prompt/failure policy; clients never receive these secrets.
I.models: `model=qwen3-asr-1.7b` selects resident Qwen3-ASR when `ASR_QWEN_ENABLED=1`; all other legacy model values select Whisper. `ASR_QWEN_*` configures checkpoint, interpreter, and GPU placement.
I.catalog: Authenticated `GET /v1/audio/models` lists transcription model IDs in OpenAI list format; `GET /v1/models` keeps its broader compatibility catalog.

§V
V1: Audio processing order is fixed and tested: validate request → transcribe with optional `language`/`prompt` context → optional conservative cleanup → optional explicitly requested translation → response.
V2: Without explicit translation request, output language remains source language and cleanup never translates.
V3: `prompt` vocabulary/context reaches Whisper `initial_prompt`, is capped, is never logged verbatim, and does not persist across requests.
V4: No generic edit-distance/fuzzy replacement rewrites transcript tokens. Terminology correction uses request context and/or context-aware service post-processing.
V5: Filler policy executes at most once in service pipeline; desktop clients receive already-final semantic text.
V6: Optional post-processing failure returns raw Whisper text; required post-processing failure returns sanitized non-2xx response without fabricated transcript.
V7: Translation endpoint/extension has explicit source/target behavior, authentication, timeout, failure policy, and tests before any client exposes translation UI.
V8: Existing transcription clients sending only current OpenAI-compatible fields retain current behavior.
V9: Tokens, prompts, vocabulary context, and transcript bodies are absent from default logs; diagnostics contain lengths/timings/status only.
V10: Spoken technical separators align only matching raw/corrected word sequences, accept Unicode letters/digits, preserve unrelated separators, and lowercase joined identifiers.
V11: Clear Spanish and English technical cues reconstruct spoken hyphen and underscore commands through the full cleanup pipeline, including space-only LLM output, while ordinary-language uses remain unchanged.
V12: Valid bilingual list controls format Markdown without inventing a title: `lista de ítems` / `item list` creates bullets; `lista numerada` / `numbered list` creates numbering; `nuevo ítem` / `new item` separates either type; the immediately preceding clause is preserved as an introduction with an implicit colon.
V13: List formatting preserves source language plus every item's content, quantity, and order; control phrases disappear only in valid list context, ordinary-language uses remain prose, and comma-separated items are inferred only for clear enumerations.
V14: Cleanup rejects substantial word replacement for every nonempty transcript, including one-to-three-word phrases; existing punctuation, technical-literal, and Spanish marker repairs still pass.
V15: Enabled Qwen selector routes audio to resident Qwen worker; `language` and bounded `prompt` reach Qwen language/context inputs, Qwen stays loaded beside Whisper, and disabled selector returns 400.
V16: Qwen worker inherits device/cache configuration but never inherits Whisper's `LD_LIBRARY_PATH`; PyTorch resolves its own compatible CUDA/cuDNN libraries.
V17: ASR catalog lists Whisper and only enabled optional transcription engines, using their configured IDs; it excludes the cleanup LLM, requires bearer auth, and does not load models.

§T
id|status|task|detail|cites
T1|.|Document semantic ownership|Document audio endpoint response as final transcript and service ownership of vocabulary, cleanup, filler policy, correction, post-processing, and translation.|G1,G2,V5,V8
T2|.|Vocabulary context contract|Keep bounded `prompt` → Whisper `initial_prompt`; document recommended short same-language terminology context; test cap, forwarding, request isolation, and log redaction.|I.transcribe,V3,V9
T3|.|Unified cleanup pipeline|Use one internal cleanup path for audio and text endpoints; define filler/correction policy, prevent duplicate passes, and test fixed stage order plus fail-open/required behavior.|I.postprocess,V1,V2,V4,V5,V6
T4|.|Translation contract|Implement and document `POST /v1/audio/translations` for English or explicitly version arbitrary-target extension; keep translation disabled absent explicit request; test auth, language behavior, timeout, and failures.|I.translate,V1,V2,V7,V8
T5|.|Compatibility matrix|Test legacy transcription request, language hint, prompt context, post-processing disabled/enabled/unavailable, and explicit translation without GPU/network/live models.|C7,V1,V2,V3,V6,V7,V8
T6|x|Bilingual technical separators|Add symmetric Spanish/English technical cues, recover spoken hyphens/underscores from space-only cleanup output, and preserve ordinary-language uses.|I.postprocess,C3,C7,V10,V11
T7|x|Bilingual formatted lists|Add explicit bullet/numbered list controls, shared item separator, implicit introduction colon, conservative comma inference, output validation, and offline bilingual regression tests.|I.postprocess,C3,C7,V12,V13
T8|x|Optional Qwen3-ASR engine|Route model selector to resident Qwen worker, forward language/context, preserve Whisper default, document isolated runtime and verify on host.|I.models,C8,V15
T9|x|ASR model catalog|Expose authenticated ASR-only model IDs so clients discover enabled engines without hardcoded choices.|I.catalog,G4,V17

§B
id|date|cause|fix
B1|2026-07-13|ASCII-only guard rejected accented spoken identifiers; transcript-wide underscore repair rewrote unrelated hyphens|V10
B2|2026-07-30|Prompt omitted `barra baja`; technical cues lacked bilingual symmetry; repair missed space-only LLM output; safety guard counted `baja` as content|V11
B3|2026-09-23|Meaningful-word guard skipped phrases shorter than four words, allowing cleanup to translate short English transcripts; applying guard before Spanish marker restoration dropped valid punctuation|V14
B4|2026-09-23|Qwen worker inherited Whisper's cuDNN library path and failed GPU inference with `GET was unable to find an engine to execute this computation`|V16
