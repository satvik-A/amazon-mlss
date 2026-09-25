"""Indic scripts -> Latin, dependency-free.
The 9 major Brahmic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share one layout: the same offset inside each 0x80 block is the same sound,
so one offset table transliterates all of them. Used only as the FALLBACK for words missing from the
learned Indic lexicon (artifacts.fit_indic_lexicon)."""
VOW = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e",
       0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "rri", 0x61: "lli"}
CON = dict(zip(range(0x15, 0x3A), "k kh g gh ng ch chh j jh ny t th d dh n t th d dh n n p ph b bh m y r r l l l v sh sh s h".split()))
CON.update({0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"})
MAT = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "rri", 0x45: "e", 0x46: "e", 0x47: "e",
       0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "lli"}
SIGN = {0x01: "n", 0x02: "n", 0x03: "h"}
VIRAMA, NUKTA = 0x4D, 0x3C
INDIC_RE = r"[ऀ-ൿ]"


def translit(s: str) -> str:
    if not s:
        return s
    out, pend = [], False  # pend: a consonant is waiting for its inherent 'a'
    for ch in s:
        o = ord(ch)
        off = (o & 0x7F) if 0x0900 <= o < 0x0D80 else None
        if off is None:  # leaving the word: drop the word-final inherent 'a' (schwa deletion)
            pend = False
            out.append(ch)
            continue
        if off in CON:
            if pend:
                out.append("a")
            out.append(CON[off]); pend = True
        elif off in MAT:
            out.append(MAT[off]); pend = False
        elif off == VIRAMA:
            pend = False
        elif off == NUKTA:
            if out and out[-1] == "j":
                out[-1] = "z"
        elif off in VOW:
            if pend:
                out.append("a"); pend = False
            out.append(VOW[off])
        elif off in SIGN:
            if pend:
                out.append("a"); pend = False
            out.append(SIGN[off])
        elif 0x66 <= off <= 0x6F:
            if pend:
                out.append("a"); pend = False
            out.append(str(off - 0x66))
        elif pend:
            out.append("a"); pend = False
    return "".join(out)
