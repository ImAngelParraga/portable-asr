# Service-Owned Transcript Semantics Spec

§G
G1: Make ASR service sole owner of transcription semantics: vocabulary biasing, cleanup, filler handling, correction, and optional translation.
G2: Return final semantic transcript from audio endpoint so thin clients only present/insert received text.
G3: Select Whisper, optional Qwen3-ASR, or optional Parakeet Ultra per request through existing OpenAI-compatible `model` field.
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
I.history: Optional `ASR_AUDIO_HISTORY_*` settings enable local diagnostic records; successful archived audio responses add `request_id`, failed archived requests return `X-Request-ID`. No HTTP archive access.
I.parakeet: `model=parakeet-ultra` selects isolated worker only when `ASR_PARAKEET_ENABLED=1`; `ASR_PARAKEET_BACKEND=photon|cpp` selects Photon or local C-API library + GGUF. Other `ASR_PARAKEET_*` settings configure ID, interpreter, GPU, cache, residency, and optional local cleanup GPU handoff. Language detection automatic; vocabulary hints and temperature ignored.
I.residency: `ASR_SINGLE_RESIDENT_MODEL=1` serializes ASR inference and unloads other ASR engines before the selected engine runs; enable keep-warm on persistent engines. Cleanup residency independent.

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
V10: Spoken technical separators align only matching raw/corrected word sequences, accept Unicode letters/digits, preserve unrelated separators, and lowercase joined identifiers except preserved uppercase acronym prefixes in numeric codes (V37).
V11: Clear Spanish and English technical cues reconstruct spoken hyphen and underscore commands through the full cleanup pipeline, including space-only LLM output, while ordinary-language uses remain unchanged.
V12: Valid bilingual list controls format Markdown without inventing a title: `lista de ítems` / `item list` creates bullets; `lista numerada` / `numbered list` creates numbering; `nuevo ítem` / `new item` separates either type; the immediately preceding clause is preserved as an introduction with an implicit colon.
V13: List formatting preserves source language plus every item's content, quantity, and order; control phrases disappear only in valid list context, ordinary-language uses remain prose, and comma-separated items are inferred only for clear enumerations.
V14: Cleanup rejects substantial word replacement for every nonempty transcript, including one-to-three-word phrases; existing punctuation, technical-literal, and Spanish marker repairs still pass.
V15: Enabled Qwen selector routes audio to resident Qwen worker; `language` and bounded `prompt` reach Qwen language/context inputs, Qwen stays loaded beside Whisper, and disabled selector returns 400.
V16: Qwen worker inherits device/cache configuration but never inherits Whisper's `LD_LIBRARY_PATH`; PyTorch resolves its own compatible CUDA/cuDNN libraries.
V17: ASR catalog lists Whisper and only enabled optional transcription engines, using their configured IDs; it excludes the cleanup LLM, requires bearer auth, and does not load models.
V18: GPU arbitration recognizes resident Whisper and Qwen workers as service-owned processes; neither blocks cleanup LLM startup after transcription, while unrelated GPU jobs remain external.
V19: Qwen decodes accepted AAC/M4A and WAV uploads to mono 16 kHz PCM WAV before model inference, keeps decoded audio only for that request, and never calls the model after decoder failure; Whisper routing is unchanged.
V20: Cleanup accepts digit formatting of equivalent spoken Spanish and English cardinal numbers through 999, including dotted sequences and zero components; it rejects changed numeric values and changed surrounding words.
V21: Explicit `deletreo ... fin de deletreo` / `spell ... end spelling` controls join Spanish or English letter names and single-letter transcripts into one word without consuming following speech; unclear or incomplete controls remain literal, spelled words survive cleanup, and block capitals become ordinary word case.
V22: List controls accept documented bullet/numbered aliases and next-item aliases while preserving item content, order, and introduction.
V23: Cleanup rejects invented words in short transcripts as well as lost words; sentence-leading Spanish `pregunta` deterministically produces `¿...?` punctuation around the original question content even when the LLM adds words or emits a statement.
V24: Audio history defaults off; enabled history preserves exact uploaded bytes, raw ASR text, final text, request/model metadata, and failed-request status under one server-generated ID; tokens, prompt content, filenames, and transcript bodies remain absent from logs.
V25: History directories use 0700 and files use 0600; finite positive count/age/byte limits evict oldest records on writes, startup, and periodic idle cleanup. Storage failures never fail transcription; invalid/unauthorized uploads create no records.
V26: Cleanup joins consecutive spoken Spanish or English single digits after an explicit hyphen/underscore in a clear technical code or isolated literal, preserving digit order and leading zeros; unchanged LLM output receives the same repair, altered digits or surrounding words are rejected, and ordinary counts remain unchanged.
V27: In clear technical literals with explicit separators, cleanup joins single-letter/spoken-digit/single-letter prefixes such as `b dos b` or `b two b` into `b2b` before safety comparison and separator repair; digit order, leading zeros, suffix words, and surrounding content are preserved, altered identifiers are rejected, and ordinary prose remains literal.

V28: Explicit project/proyecto context reconstructs spoken technical separators through full cleanup even when the LLM preserves spoken commands or emits spaces; ordinary prose remains literal and in-sentence technical cue casing follows source.
V29: Explicit class/clase followed by two to four name words ending in a conventional class-role suffix and a clear boundary joins PascalCase before LLM and safety comparison; exact source words and repetitions are protected after cleanup, inline-code wrappers are removed only around recognized names, existing identifiers and explicit separators remain intact, and ambiguous names or prose remain literal.
V30: Sentence-leading Spanish `pregunta` controls at transcript start, after sentence punctuation, or after line breaks deterministically mark only their following sentence as `¿...?`, including repeated commands and a period after the marker; ordinary in-sentence uses remain literal, model punctuation cannot expand the question boundary, and question words remain unchanged.
V31: Default cleanup instructions distinguish direct Spanish questions and conversational requests from indirect questions and ambiguous statements, ignore unreliable ASR periods as evidence of intent, preserve words, and include positive and negative examples; explicit indirect-question and ability-explanation prefixes reject invented interrogative punctuation or word loss unless source punctuation or a control establishes a question; offline tests verify behavior without a live model.
V32: Enabled successful cleanup deterministically formats explicit Spanish/English euro, dollar, and percent amounts as symbols, including unchanged LLM output; safety comparison preserves currency/unit identity, amount precision, and surrounding words while accepting equivalent currency prefix/suffix forms. Ordinary currency mentions and unsupported spoken amounts remain literal; question and list formatting retain the complete quantity. Disabled/unavailable cleanup keeps its existing raw-text failure contract.

V33: Embedded English `hyphen`/`underscore` between name words deterministically formats lowercase identifiers without a nearby technical cue; ordinary punctuation mentions and ambiguous multiword class names remain literal. Prepare recognized separators before LLM cleanup so safety fallbacks retain the identifier; disabled/unavailable cleanup preserves its raw-text contract.

V34: Successful enabled cleanup prepares bounded adjacent preposition restarts marked by comma, ellipsis, or pause dash before model and safety comparison: a/al, de/del, on/in, in/on, and repeated prepositions retain the final choice and following content, including unchanged model output and numeric phrases. Alternatives, enumerations, quotations, code, spelling/list controls, meaningful corrections, and unmarked or ambiguous starts remain literal; disabled/unavailable cleanup retains raw-text failure contracts.

V35: Bounded Spanish explanatory/continuation clauses beginning with unaccented `que` plus no/ya/time cues, optionally after disculpa/perdona/perdón, remain declarative through cleanup; invented question punctuation and the explanatory `qué` accent are repaired while embedded interrogative accents and independent questions survive. Source question punctuation or an explicit pregunta control takes precedence; direct interrogative `qué` remains eligible for question formatting.
V36: Cleanup preserves existing Spanish haber conjugations, including person, tense, and mood. Changed forms are restored locally only when all surrounding normalized words align exactly; otherwise prepared source is returned. Accent/case formatting and a safe missing auxiliary insertion when no existing haber form is present retain their existing behavior.

V37: Enabled successful cleanup concatenates two or more separately spoken Spanish single-word cardinal or digit chunks after an explicit code separator, including mixed written chunks, before LLM and safety comparison; task/tarea/ticket/issue cues establish code context. Preserve chunk order, leading zeros, and source uppercase acronym prefixes in numeric codes, including existing codes and model-lowercased output; reject changed digits, prefixes, or surrounding words. Ordinary counts, conjunction-based or unsupported cardinals remain literal; disabled/unavailable cleanup retains raw failure behavior.

V38: Parakeet selector defaults disabled, returns 400 when disabled, appears in authenticated catalogs only when enabled, preserves default/legacy Whisper and Qwen routing, and uses existing cleanup/history contracts without client changes.
V39: Parakeet worker decodes WAV/AAC/M4A to request-local mono 16 kHz PCM, removes temporary audio on success/failure, never forwards unsupported language/prompt/temperature options, and returns sanitized failures without audio/transcript contents in logs.
V40: Parakeet worker uses isolated Python/CUDA libraries and explicit GPU/cache configuration, overriding inherited Hugging Face hub cache aliases; optional shared-GPU mode stops local cleanup before inference and terminates nonresident worker before cleanup. Resident worker counts as service-owned; shutdown/failure cleans worker resources.
V41: Cleanup shutdown reaps owned children and treats Linux zombie PIDs as exited, so GPU handoff does not wait for the stop timeout after the process already released its resources.

V42: C++ Parakeet uses configured local library/GGUF, owns one reusable C-API context, frees returned strings on decode success/failure, and frees context on close. Its worker sees only its runtime library directory, never Whisper's libraries; Photon remains the default backend.
V43: Single-resident mode unloads only unselected ASR engines before inference, retains the selected warm worker through cleanup and repeated requests, serializes switches against in-flight inference, and preserves legacy independent-worker behavior when disabled. Same auth/catalog/history/cleanup contracts apply to C++ Parakeet.

V44: GPU process watcher excludes the service and its descendant workers/CPU audio decoders by parent ancestry; unrelated matching executables remain watched. Missing/malformed/racing process metadata never crashes the watcher.
V45: Concurrency admission awaits available capacity without periodic polling, retains configured limits and sanitized 503 timeout behavior, and cancellation/timeouts never consume or leak slots. Initialize capacity per application lifespan/event loop.
V46: Cleanup requests use configured `ASR_POSTPROCESS_REPEAT_PENALTY` (legacy default 1.18) with neutral presence/frequency penalties; general chat retains its existing repetition policy. Cache, thinking and failure contracts remain intact.
V47: Local cleanup startup forwards opt-in `ASR_LLM_SPEC_TYPE` to llama-server; default none omits the flag for legacy-runtime compatibility. Copy speculation uses the existing model and preserves configured GPU placement.
V48: Opt-in cleanup edit validation deterministically rejects changed known Spanish conjugations sharing a lemma, lost valid verb accents and altered negation sequences. Bundled attributed conjugation data loads once without GPU/network; missing data retains prepared source. Other changed content receives one bounded same-model A/B audit; reject, invalid verdict or audit failure retains prepared source before question repair. Case/punctuation and equivalent numeric formatting skip audit; safe spelling/function-word repairs remain eligible. Local audits constrain output grammar and log no text.

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
T10|x|Decode Qwen client audio|Normalize uploaded AAC/M4A and WAV through FFmpeg before Qwen inference, remove temporary PCM after each request, and cover failure without GPU or model downloads.|I.transcribe,I.models,V19
T11|x|Protect numeric formatting|Canonicalize equivalent spoken cardinals for cleanup safety checks and test version numbers, surrounding words, and wrong values.|I.postprocess,V20
T12|x|Join explicitly spelled words|Prepare delimited Spanish/English spelling before cleanup, protect letter order and case, and test unknown or incomplete controls.|I.postprocess,V21
T13|x|Expand spoken list controls|Accept bullet/numbered and next-item voice aliases, document the syntax, and test output structure.|I.postprocess,V22
T14|x|Retain recent diagnostic audio|Add optional bounded local audio history, raw/final text and failure metadata, request IDs, idle expiry, private permissions, and offline regression tests.|I.history,I.transcribe,V8,V9,V24,V25

T15|x|Repair class and project literals|Recognize bilingual project cues; prepare and protect bounded PascalCase class-role names; test full cleanup, source cue case, and ambiguous prose.|I.postprocess,C3,C7,V28,V29
T16|x|Improve Spanish question formatting|Strengthen contextual question instructions; prepare and enforce sentence-scoped question controls; verify direct/indirect questions, repeated commands, fallback, and live text samples.|I.postprocess,C3,C7,V23,V30,V31
T17|x|Format quantity symbols consistently|Prepare explicit euro/dollar/percent amounts before cleanup, compare symbol identity during safety checks, normalize unchanged model output, and include units in question boundaries; verify bilingual amounts, precision, lists, ordinary mentions, and failure contracts.|I.postprocess,I.transcribe,C3,C7,V32

T18|x|Clean local speech restarts|Prepare and validate bilingual preposition restarts; preserve meaningful corrections, alternatives, literal contexts, and raw failure behavior; verify model corrections and unchanged output offline.|I.postprocess,I.transcribe,C3,C7,V34

T19|x|Preserve explanatory que and haber forms|Protect bounded explanatory clauses, retain explicit/direct questions and embedded accents, restore aligned haber conjugations, and verify reported text/audio cases plus offline regressions.|I.postprocess,I.transcribe,C3,C7,V35,V36

T20|x|Preserve numeric code chunks and acronym case|Prepare separate Spanish cardinal/digit chunks before cleanup; recognize ticket cues; preserve acronym case; test unchanged, numeric, spaced, wrong-value, leading-zero, and ordinary-prose output.|I.postprocess,I.transcribe,C3,C7,V26,V37

T21|x|Optional Parakeet Ultra engine|Add Photon worker, optional catalog/routing/config, shared-GPU handoff and deployment support; verify offline contracts plus live WAV/M4A with unchanged default.|I.parakeet,I.catalog,I.transcribe,V38,V39,V40,V41

T22|x|Resident Ultra on Pascal|Support C++ runtime + Ultra GGUF; keep last selected ASR resident on one GPU, cleanup on its own GPU; verify context ownership, switching, audio formats, repeated requests, and live GPU residency.|I.parakeet,I.residency,I.transcribe,V38,V39,V42,V43

T23|x|Remove avoidable cleanup stalls|Exclude owned decoder descendants from GPU reservations; wake queued requests directly on slot release; verify external jobs, missing process metadata, prompt wakeup, timeout and cancellation.|I.transcribe,I.postprocess,V44,V45
T24|x|Configure cleanup repetition penalty|Expose cleanup-only repetition setting so deployed model profiles reproduce measured copy-preserving behavior; verify request payload isolation from general chat.|I.config,I.postprocess,C3,C7,V46
T25|x|Enable optional copy speculation|Expose local llama-server speculative type, preserve legacy default, deploy measured ngram-mod profile and verify live output parity plus ASR/cleanup residency.|I.config,I.postprocess,V43,V47
T26|~|Validate meaning-changing cleanup edits|Keep existing protections; audit changed wording and potentially meaningful accents with resident model, preserve failure fallback and formatting, test tense/person regressions and legitimate corrections, measure and deploy.|I.config,I.postprocess,C3,C7,V9,V48

§B
id|date|cause|fix
B1|2026-07-13|ASCII-only guard rejected accented spoken identifiers; transcript-wide underscore repair rewrote unrelated hyphens|V10
B2|2026-07-30|Prompt omitted `barra baja`; technical cues lacked bilingual symmetry; repair missed space-only LLM output; safety guard counted `baja` as content|V11
B3|2026-09-23|Meaningful-word guard skipped phrases shorter than four words, allowing cleanup to translate short English transcripts; applying guard before Spanish marker restoration dropped valid punctuation|V14
B4|2026-09-23|Qwen worker inherited Whisper's cuDNN library path and failed GPU inference with `GET was unable to find an engine to execute this computation`|V16
B5|2026-09-23|GPU arbitration exempted the resident Whisper worker but classified the resident Qwen worker as external; required cleanup could not start after Qwen transcription and returned HTTP 500|V18
B6|2026-09-23|Qwen forwarded Android AAC/M4A to librosa 1.0 and libsndfile, which cannot decode that container and raised LibsndfileError|V19
B7|2026-09-25|Cleanup safety guard compared spoken number words against digits as unrelated tokens and restored raw transcript even when LLM produced equivalent 4.58.0|V20
B8|2026-09-25|English spelling control existed but parser recognized only literal letters and Spanish letter names; spoken English names remained unjoined|V21
B9|2026-09-28|Meaningful-word guard checked only retained input words, so cleanup could insert `por qué` into a short question; leading `pregunta` punctuation depended on LLM output|V23
B10|2026-09-30|Safety guard compared individually spoken code digits with a joined number as unequal content and discarded the entire correction, including the explicit hyphen|V26
B11|2026-09-30|Technical parser started at the last letter of a spaced alphanumeric prefix; safety guard treated `b dos b` and `b2b` as unequal content and restored unformatted table names|V27
B12|2026-10-02|Technical cue list omitted project/proyecto and restored explicitly spoken hyphens; class-role names had no deterministic joining and cleanup capitalized cue words|V28,V29
B13|2026-10-02|Live cleanup wrapped identifiers in inline code; class protection rejected equivalent formatting and restored prepared text with spoken separators|V28,V29
B14|2026-10-05|Question instructions lacked contextual direct/indirect examples; deterministic pregunta enforcement checked only transcript start and trusted model sentence boundaries|V30,V31
B15|2026-10-05|Question enforcement treated decimal points as sentence ends; live LLM ignored negative prompt examples and turned ability explanations or dime requests into direct questions|V30,V31
B16|2026-10-06|Safety comparison discarded currency symbols and counted currency words as lost content, restoring spoken amounts; deterministic question boundaries excluded trailing quantity symbols|V32
B17|2026-10-08|Separator confidence required a technical cue for embedded word-only identifiers, restoring spoken commands and allowing name capitalization; unprepared safety fallbacks could restore commands|V33
B18|2026-10-08|Cleanup prompt preserved abandoned prepositions; word-retention guard rejected deletion in short or numeric phrases, and unchanged model output had no restart repair|V34
B19|2026-10-14|Question protections omitted explanatory que clauses; broad word-retention threshold allowed single haber conjugation changes, turning correct ASR statements into questions with changed person or mood|V35,V36
B20|2026-10-21|Code repair accepted only single spoken digits and missed ticket cues; mixed cardinal chunks reached LLM, lost a zero, and failed safety comparison; generic identifier casing lowercased acronym prefixes|V37
B21|2026-10-29|New worker test used invalid grouped context-manager syntax; corrected mechanical typo, no new behavior invariant needed|V40
B22|2026-10-29|Cleanup stop treated unreaped zombie PID as live and waited 20 seconds during shared-GPU handoff; inherited hub cache alias also overrode the Parakeet cache path|V40,V41
B23|2026-10-29|GPU watcher matched service-owned CPU FFmpeg decoders as external work and could stop cleanup with a ten-second GPU reservation|V44
B24|2026-10-29|Concurrency admission polled a threading semaphore every 250 ms, delaying queued requests after capacity was already free|V45
B25|2026-10-29|Word-retention threshold accepted small tense/person substitutions in otherwise preserved prose; conjugation protection covered only haber|V48
B26|2026-10-29|One-token audit accepted a present-to-conditional substitution and removed repeated negation; model judgment alone cannot enforce conjugation/negation invariants|V48
B27|2026-10-29|Pinned conjugation data omitted common lemmas including tener and escribir; regression coverage required supplemental paradigms|V48
