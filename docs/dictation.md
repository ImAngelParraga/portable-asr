# Dictation formatting

These controls are available with the built-in cleanup pipeline when post-processing is enabled. They are optional. Ordinary dictation needs no spoken commands.

## Speech Restarts

With post-processing enabled, bounded preposition restarts keep the final choice: `Quiero ir a, al edificio de allí` becomes `Quiero ir al edificio de allí`, and `You can see that on, in this file` becomes `You can see that in this file`. Commas, ellipses, and pause dashes are accepted between adjacent prepositions. Supported repairs are `a` → `al`, `de` → `del`, `on` ↔ `in`, and repeated prepositions followed by phrase content. Repairs run before model cleanup and safety comparison, and also handle unchanged model output.

Meaningful corrections such as `tomorrow, no, Friday`, emphasis, alternatives, quoted speech, code, and explicit spelling/list controls remain outside this rule. Unmarked or ambiguous restarts stay literal. The cleanup model receives text, so it cannot recover pauses that ASR omitted. Disabled or unavailable optional cleanup returns the original raw transcript.

## Questions In Dictation

Default post-processing uses sentence context to restore Spanish direct questions and conversational requests, even when the ASR output ends with a period. For example, `puedes enviarme el archivo` normally becomes `¿Puedes enviarme el archivo?`. Indirect questions such as `no sé cómo funciona esto` remain statements. Cleanup changes punctuation and accents without inserting question words.

For an ambiguous sentence, say `pregunta` at its beginning: `ya terminé. pregunta puedes revisarlo. luego hablamos.` becomes `Ya terminé. ¿Puedes revisarlo? Luego hablamos.` The control works at transcript start, after sentence punctuation, or after a line break. Each control applies only to its following sentence; repeated controls work independently. An ASR period immediately after the marker is accepted, and decimal points inside the question remain intact. Ordinary in-sentence uses of `pregunta` remain literal.

Question controls are prepared before LLM cleanup and their boundaries are enforced afterward. If model output changes the words or merges the controlled sentence with surrounding speech, the service restores the prepared source text conservatively. Disabled or unavailable optional post-processing retains its existing raw-text fallback.

The cleanup LLM receives text, not audio. It cannot recover intonation alone: an ambiguous `mañana vienes` stays declarative unless context or a control establishes a question. Automatic question formatting is model dependent; the explicit control provides a deterministic fallback. A bounded set of Spanish indirect-question and ability-explanation prefixes, including `no sé`, `dime cómo`, `con este permiso`, and `gracias a`, is protected against invented questions. Existing question punctuation or an explicit control overrides that protection.

## Spoken Formatted Lists

When post-processing is enabled, bilingual voice controls can create Markdown lists:

- `lista de ítems` / `item list`: start a bulleted list.
- `lista numerada` / `numbered list`: start a numbered list.
- `nuevo ítem` / `new item`: start the next item in either list type.
- `fin de lista` / `end of list`: optionally end the list before continuing with normal prose.

Alternatives: `lista con viñetas` or `bullet list` for bullets, `lista con números` for numbering, and `nuevo elemento`, `siguiente elemento`, or `siguiente ítem` for the next item. English `next item` also works. State at least two items. Each item may contain commas when you use an explicit next-item phrase. Say `fin de lista` before continuing normal dictation.

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
