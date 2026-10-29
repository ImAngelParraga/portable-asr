# Spanish conjugation lookup

`spanish-verbs.json.gz` contains 6,703 lemmas derived from UniMorph Spanish
and its English Wiktionary source contributors, plus the supplemental
tables in `spanish-verbs-extra.json`. The pinned source supplied 6,695
lemmas and 381,195 single-word forms; it omitted common verbs including
tener and escribir. The eight supplemental paradigms fill those known gaps.
The service uses exact forms, including accents, to recognize a proposed
change between conjugations of the same verb. It is not a contextual parser
and does not infer an unrecognized source form's tense.

Source: <https://github.com/unimorph/spa>

Pinned revision: `b9655efb0e5ce0f99809badaa7f4e93017ab8430`

Original `spa` file SHA-256:
`7a165d4b44eb078d23964f3eee3813f5140c5df0ffab0143c806c562e67f8771`

The derived data remains licensed under **Creative Commons
Attribution-ShareAlike 3.0 Unported**, independently of the service code:
<https://creativecommons.org/licenses/by-sa/3.0/>.

Changes: retain `V` rows with alphabetic lemma/form and no `PRO` feature;
normalize Unicode NFC and lowercase; deduplicate forms per lemma; omit
feature labels; merge supplemental paradigms; sort and gzip the JSON with a zero timestamp. No source
data is fetched during installation or inference.

Regenerate from the pinned `spa` TSV with:

```bash
python3 scripts/build-spanish-verb-lexicon.py /path/to/spa
```
