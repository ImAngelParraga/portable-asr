# Dictation formatting

These controls are available with the built-in cleanup pipeline when post-processing is enabled. They are optional. Ordinary dictation needs no spoken commands.

## Speech Restarts

With post-processing enabled, bounded preposition restarts keep the final choice: `Quiero ir a, al edificio de allí` becomes `Quiero ir al edificio de allí`, and `You can see that on, in this file` becomes `You can see that in this file`. Commas, ellipses, and pause dashes are accepted between adjacent prepositions. Supported repairs are `a` → `al`, `de` → `del`, `on` ↔ `in`, and repeated prepositions followed by phrase content. Repairs run before model cleanup and safety comparison, and also handle unchanged model output.

Meaningful corrections such as `tomorrow, no, Friday`, emphasis, alternatives, quoted speech, code, and explicit spelling/list controls remain outside this rule. Unmarked or ambiguous restarts stay literal. The cleanup model receives text, so it cannot recover pauses that ASR omitted. Disabled or unavailable optional cleanup returns the original raw transcript.

Spanish prepositional phrases can also restart: `dentro del tiempo, dentro del tempo` becomes `dentro del tempo`. This bounded rule requires a repeated prefix of two to five words, a pause separator, and a final word of at least four letters changing by one letter. It keeps the final spoken word without choosing replacement vocabulary. Verbs, quantities, identical emphasis, enumerations, quoted or metalinguistic mentions, and unrelated word changes remain literal.

## Questions In Dictation

Default post-processing uses sentence context to restore Spanish direct questions and conversational requests, even when the ASR output ends with a period. For example, `puedes enviarme el archivo` normally becomes `¿Puedes enviarme el archivo?`. Indirect questions such as `no sé cómo funciona esto` remain statements. Cleanup changes punctuation and accents without inserting question words.

For an ambiguous sentence, say `pregunta` at its beginning: `ya terminé. pregunta puedes revisarlo. luego hablamos.` becomes `Ya terminé. ¿Puedes revisarlo? Luego hablamos.` The control works at transcript start, after sentence punctuation, or after a line break. Each control applies only to its following sentence; repeated controls work independently. An ASR period immediately after the marker is accepted, and decimal points inside the question remain intact. Ordinary in-sentence uses of `pregunta` remain literal.

Question controls are prepared before LLM cleanup and their boundaries are enforced afterward. If model output changes the words or merges the controlled sentence with surrounding speech, the service restores the prepared source text conservatively. Disabled or unavailable optional post-processing retains its existing raw-text fallback.

The cleanup LLM receives text, not audio. It cannot recover intonation alone: an ambiguous `mañana vienes` stays declarative unless context or a control establishes a question. Automatic question formatting is model dependent; the explicit control provides a deterministic fallback. A bounded set of Spanish indirect-question and ability-explanation prefixes, including `no sé`, `dime cómo`, `con este permiso`, and `gracias a`, is protected against invented questions. Existing question punctuation or an explicit control overrides that protection.

## Spoken Formatted Lists

When post-processing is enabled, bilingual voice controls can create Markdown lists:

- `lista de ítems` / `item list`: start a bulleted list.
- `lista numerada` / `lista ordenada` / `numbered list`: start a numbered list.
- `nuevo ítem` / `new item`: start the next item in either list type.
- `fin de lista` / `end of list`: optionally end the list before continuing with normal prose.

Alternatives: `lista con viñetas` or `bullet list` for bullets, `lista con números` for numbering, and `nuevo elemento`, `siguiente elemento`, or `siguiente ítem` for the next item. English `next item` also works. State at least two items. Each item may contain commas when you use an explicit next-item phrase. Say `fin de lista` before continuing normal dictation.

You may also say `nuevo elemento` before the first item. Sentence punctuation inserted by recognition after a control is accepted: `Lista ordenada. Nuevo elemento. Abrir el archivo. Nuevo elemento. Guardar los cambios.` produces two numbered items, preserving each item's sentence punctuation.

The clause before the list control becomes the introduction with an implicit colon. For example, `hoy tengo que comprar lista de ítems plátanos nuevo ítem tomates` becomes:

```markdown
Hoy tengo que comprar:

- Plátanos
- Tomates
```

Clear comma-separated short items also work. Use `nuevo ítem` / `new item` when an item contains commas or item boundaries could be ambiguous.

For a difficult word, say `deletreo`, speak letter names, then say `fin de deletreo`. Example: `escribe deletreo eme a erre i a fin de deletreo aquí` becomes `Escribe maria aquí.` In English, say `spell M A R I A end spelling` or `spell em ay ar eye ay end spelling`; both produce `maria`. The service joins the letters into one word and uses normal case. It can add an accent when context makes it clear; say the accent explicitly when exact spelling matters. The closing phrase is required so following words remain separate. Unknown letter names leave the span unchanged.

## Spoken Class And Project Names

With post-processing enabled, `class task controller` / `clase task controller` becomes `class TaskController` / `clase TaskController`. This conservative rule accepts two to four name words ending in `Controller`, `Manager`, `Handler`, `Factory`, `Builder`, `Adapter`, `Provider`, or `Repository`, followed by punctuation, sentence end, or a clear clause boundary. Existing identifiers, explicit separators, and ambiguous prose remain unchanged. Class names are prepared before cleanup and protected against LLM replacement.

For project names, say separators explicitly: `project demo hyphen tools` becomes `project demo-tools`; `proyecto demo guion bajo tools` becomes `proyecto demo_tools`. Technical cue words remain ordinary nouns inside a sentence. Exact unconventional spelling still requires vocabulary context or spelling controls.

## Terminal Command Dictation

Start a recording with `comando` (also `modo comando` or English `command`).
The rest of that recording becomes command text through deterministic rules,
without LLM cleanup or changes to the service prompt. This works even when
optional post-processing is disabled. Every following recording starts in normal
dictation again. The prefix is reserved only at the beginning of a recording.

| Say | Insert |
|---|---|
| `comando barra new` | `/new` |
| `comando slash clear` | `/clear` |
| `comando deletreo eme ka de i erre` | `mkdir` |
| `comando deletreo m k d i r fin deletreo espacio proyectos` | `mkdir proyectos` |
| `comando git espacio guion guion version` | `git --version` |
| `comando touch espacio mi guion bajo archivo punto txt` | `touch mi_archivo.txt` |
| `command spell em kay dee eye ar end spelling space projects` | `mkdir projects` |

Normal words are separated with spaces. Spoken symbols attach to adjacent text;
use `espacio` / `space` when a space is needed beside a symbol. Supported symbols:
`barra` / `slash`, `barra invertida` / `backslash`, `guion` / `hyphen` / `dash`,
`guion bajo` / `barra baja` / `underscore`, `punto` / `dot`, `dos puntos` / `colon`,
`igual` / `equals`, `comilla simple` / `single quote`, `comilla doble` / `comillas` /
`double quote`, `pipe` / `tubería`, `ampersand`, `mayor que` / `greater than`,
`menor que` / `less than`, `dólar` / `dollar`, `asterisco` / `asterisk`, and `tilde`.
Only spoken symbol names guarantee literal punctuation: ASR sentence punctuation
is removed. No newline or Enter is generated.

Commands default to lowercase. `mayúscula` / `uppercase` uppercases the next word,
or the next letter inside spelling. For example, `comando echo espacio dólar
mayúscula home` inserts `echo $HOME`. To spell a reserved word, use its letter names.
Spelling accepts Spanish and English letter names, single letters, and individual
digits (including spoken zero through nine). Close a spelling span with `fin
deletreo`, `fin de deletreo`, or `end spelling` before continuing the command; the
end of a recording also closes it. Unknown letter names or incomplete controls
return HTTP 422 rather than guessing a command. Recognition errors remain possible;
check the inserted text before pressing Enter.

Successful command audio responses add `"verbatim": true` beside `text`.
OpenVoiceIME uses this boolean to suppress its optional trailing space, including
when insertion is confirmed or retried later. Other clients must preserve command
text exactly themselves. Normal dictation responses keep their existing shape.
