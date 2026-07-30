# Service-Owned Transcript Semantics Spec

§G
G1: Make ASR service sole owner of transcription semantics: vocabulary biasing, cleanup, filler handling, correction, and optional translation.
G2: Return final semantic transcript from audio endpoint so thin clients only present/insert received text.

§C
C1: Keep `POST /v1/audio/transcriptions` OpenAI-compatible for existing clients.
C2: No client-specific vocabulary, fuzzy threshold, filler filter, or post-processing model required in desktop client.
C3: Cleanup remains conservative, meaning-preserving, language-preserving unless translation explicitly requested.
C4: Post-processing remains optional; disabled/unavailable optional cleanup fails open to raw Whisper text unless request selects required behavior.
C5: Vocabulary context is bounded, sanitized, excluded from logs, and must not silently become permanent server state.
C6: Translation never occurs implicitly. Request must select translation and target semantics explicitly.
C7: Tests require no GPU, network, live Whisper, or live LLM.

§I
I.transcribe: `POST /v1/audio/transcriptions` accepts multipart audio, `model`, `temperature`, optional `language`, optional bounded `prompt`; returns final text after configured service processing.
I.translate: Add explicit translation API contract before enabling client controls: OpenAI-compatible `POST /v1/audio/translations` for English, with separately documented/versioned extension if arbitrary target languages are supported.
I.postprocess: `POST /v1/text/postprocess` and `POST /v1/chat/completions` expose service cleanup directly; audio endpoint may invoke same internal pipeline.
I.config: `ASR_POSTPROCESS_*` owns provider/model/token/timeout/prompt/failure policy; clients never receive these secrets.

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

§T
id|status|task|detail|cites
T1|.|Document semantic ownership|Document audio endpoint response as final transcript and service ownership of vocabulary, cleanup, filler policy, correction, post-processing, and translation.|G1,G2,V5,V8
T2|.|Vocabulary context contract|Keep bounded `prompt` → Whisper `initial_prompt`; document recommended short same-language terminology context; test cap, forwarding, request isolation, and log redaction.|I.transcribe,V3,V9
T3|.|Unified cleanup pipeline|Use one internal cleanup path for audio and text endpoints; define filler/correction policy, prevent duplicate passes, and test fixed stage order plus fail-open/required behavior.|I.postprocess,V1,V2,V4,V5,V6
T4|.|Translation contract|Implement and document `POST /v1/audio/translations` for English or explicitly version arbitrary-target extension; keep translation disabled absent explicit request; test auth, language behavior, timeout, and failures.|I.translate,V1,V2,V7,V8
T5|.|Compatibility matrix|Test legacy transcription request, language hint, prompt context, post-processing disabled/enabled/unavailable, and explicit translation without GPU/network/live models.|C7,V1,V2,V3,V6,V7,V8
T6|x|Bilingual technical separators|Add symmetric Spanish/English technical cues, recover spoken hyphens/underscores from space-only cleanup output, and preserve ordinary-language uses.|I.postprocess,C3,C7,V10,V11

§B
id|date|cause|fix
B1|2026-07-13|ASCII-only guard rejected accented spoken identifiers; transcript-wide underscore repair rewrote unrelated hyphens|V10
B2|2026-07-30|Prompt omitted `barra baja`; technical cues lacked bilingual symmetry; repair missed space-only LLM output; safety guard counted `baja` as content|V11
