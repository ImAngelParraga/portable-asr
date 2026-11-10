"""Deterministic, per-recording command dictation and shared letter names."""

import re
from collections.abc import Collection


SPOKEN_LETTERS = {
    "a": "a", "be": "b", "b": "b", "ce": "c", "c": "c", "de": "d", "d": "d",
    "e": "e", "efe": "f", "f": "f", "ge": "g", "g": "g", "hache": "h",
    "i": "i", "jota": "j", "j": "j", "ka": "k", "k": "k", "ele": "l",
    "l": "l", "eme": "m", "m": "m", "ene": "n", "n": "n", "eñe": "ñ",
    "o": "o", "pe": "p", "p": "p", "cu": "q", "q": "q", "erre": "r",
    "r": "r", "ese": "s", "s": "s", "te": "t", "t": "t", "u": "u",
    "uve": "v", "v": "v", "doble u": "w", "w": "w", "equis": "x",
    "x": "x", "ye": "y", "i griega": "y", "y": "y", "zeta": "z", "z": "z",
    "a con tilde": "á", "e con tilde": "é", "i con tilde": "í",
    "o con tilde": "ó", "u con tilde": "ú",
    "ay": "a", "bee": "b", "see": "c", "dee": "d", "ee": "e",
    "eff": "f", "gee": "g", "aitch": "h", "eye": "i", "jay": "j",
    "kay": "k", "ell": "l", "em": "m", "en": "n", "oh": "o",
    "pee": "p", "cue": "q", "ar": "r", "ess": "s", "tee": "t",
    "you": "u", "vee": "v", "double you": "w", "ex": "x",
    "why": "y", "zee": "z", "zed": "z",
}

class CommandDictationError(ValueError):
    """An explicit command cannot be converted without guessing."""


_PREFIX = re.compile(r"^\s*(?:modo\s+comando|comando|command)\b[\s.,:;!?¡¿]*", re.IGNORECASE)
_SYMBOLS = {
    "barra": "/", "slash": "/", "barra invertida": "\\", "backslash": "\\",
    "guion": "-", "guión": "-", "hyphen": "-", "dash": "-",
    "guion bajo": "_", "guión bajo": "_", "barra baja": "_", "underscore": "_",
    "punto": ".", "dot": ".", "espacio": " ", "space": " ",
    "dos puntos": ":", "colon": ":", "igual": "=", "equals": "=",
    "comilla simple": "'", "single quote": "'", "comillas": '"',
    "comilla doble": '"', "double quote": '"', "pipe": "|", "tubería": "|",
    "ampersand": "&", "mayor que": ">", "greater than": ">",
    "menor que": "<", "less than": "<", "dólar": "$", "dollar": "$",
    "asterisco": "*", "asterisk": "*", "tilde": "~",
}
_DIGITS = dict(zip(
    "cero uno dos tres cuatro cinco seis siete ocho nueve zero one two three four five six seven eight nine".split(),
    "01234567890123456789",
))
_SPELL_START = {"deletreo", "spell"}
_SPELL_END = ("fin de deletreo", "fin deletreo", "end spelling")
_UPPER = {"mayúscula", "mayuscula", "uppercase", "capital"}


def _match_phrase(tokens: list[str], index: int, phrases: Collection[str]) -> tuple[str, int] | None:
    for size in (3, 2, 1):
        phrase = " ".join(tokens[index:index + size])
        if len(tokens[index:index + size]) == size and phrase in phrases:
            return phrase, size
    return None


def _spelled_command_word(tokens: list[str], index: int) -> tuple[str, int]:
    letters: list[str] = []
    uppercase = False
    while index < len(tokens):
        end = _match_phrase(tokens, index, _SPELL_END)
        if end:
            index += end[1]
            break
        token = tokens[index]
        if token in _UPPER:
            if uppercase:
                raise CommandDictationError("Repeated uppercase control in command spelling.")
            uppercase = True
            index += 1
            continue
        match = _match_phrase(tokens, index, SPOKEN_LETTERS)
        if match:
            letter = SPOKEN_LETTERS[match[0]]
            index += match[1]
        elif token in _DIGITS or (len(token) == 1 and token.isdigit()):
            letter = _DIGITS.get(token, token)
            index += 1
        else:
            raise CommandDictationError("Unrecognized command spelling. Use letter names and fin deletreo before continuing.")
        letters.append(letter.upper() if uppercase else letter)
        uppercase = False
    if not letters or uppercase:
        raise CommandDictationError("Incomplete command spelling.")
    return "".join(letters), index


def parse_spoken_command(text: str) -> str | None:
    """Return exact command text, None for prose, or reject malformed controls."""
    prefix = _PREFIX.match(text)
    if prefix is None:
        return None
    body = text[prefix.end():].strip()
    # Recognition punctuation separates speech; literal symbols use spoken names.
    body = re.sub(r"(?<=\w)[.,;:!?]+(?=\s|$)", "", body)
    tokens = body.lower().split()
    if not tokens:
        raise CommandDictationError("Say a command after comando.")
    output = ""
    previous_word = False
    uppercase = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _UPPER:
            if uppercase:
                raise CommandDictationError("Repeated uppercase control in command.")
            uppercase = True
            index += 1
            continue
        symbol = _match_phrase(tokens, index, _SYMBOLS)
        if symbol:
            if uppercase:
                raise CommandDictationError("Uppercase must precede a word or letter.")
            output += _SYMBOLS[symbol[0]]
            previous_word = False
            index += symbol[1]
            continue
        if _match_phrase(tokens, index, _SPELL_END):
            raise CommandDictationError("Spelling end without deletreo.")
        if token in _SPELL_START:
            word, index = _spelled_command_word(tokens, index + 1)
        else:
            word = _DIGITS.get(token, token)
            index += 1
        if uppercase:
            word = word.upper()
            uppercase = False
        if previous_word:
            output += " "
        output += word
        previous_word = True
    if uppercase or not output.strip():
        raise CommandDictationError("Incomplete command.")
    return output
