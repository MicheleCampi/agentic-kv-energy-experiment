#!/usr/bin/env python3
"""Gate for the PROTOCOL.md translation.

A translation is a transformation of prose, not of evidence. Every figure,
path, filename, section reference and identifier must survive it byte for
byte: a document whose numbers drifted during translation is worse than an
untranslated one, because the drift is invisible.

Compares the multiset of load-bearing tokens between two files and reports
what appears in one and not the other. Locale-aware on decimals: the
Italian text writes 19,01% where English writes 19.01%, so numbers are
normalised before comparison while everything else is matched literally.

Usage: check_translation.py <italian.md> <english.md>
Exit 0 when the two carry the same evidence, 1 otherwise.
"""
import collections
import re
import sys

# Numbers: 1.234,56 / 19,01 / 0,009127 / 251.320 / 25.5 / $1.29
# The lookbehind must reject A10, ADR-011, Qwen2.5 -- identifiers whose
# digits are not evidence -- without rejecting the far end of a range
# written 1,20-1,26. A stricter lookbehind that also rejected a digit
# after a hyphen turned the far end into a bare '26' -- it broke the
# very case it was added for, found by testing the extractor against
# the document instead of reasoning about it.
NUM = re.compile(r'(?<![\w/.])\$?\d[\d.,]*%?')
# Paths, filenames, dotted identifiers: main.rs:530, run_cost_cell.py,
# exp-results/..., arXiv:2605.26297, ADR-011, Fig. 7
IDENT = re.compile(
    r'\b(?:[\w./-]+\.(?:py|rs|md|sh|json|jsonl|txt)\b(?::\d+)?'
    r'|ADR-\d+'
    r'|arXiv:[\d.]+'
    r'|[A-Z]{1,3}\d(?:_\w+)?'
    r'|exp-results/[\w/-]+)')


def normalise_number(tok):
    """Make 19,01 and 19.01 the same token; keep 251.320 == 251,320."""
    t = tok.lstrip('$').rstrip('%')
    t = t.replace('.', '').replace(',', '')
    return t.lstrip('0') or '0'


def tokens(path):
    text = open(path, encoding='utf-8').read()
    nums = collections.Counter(normalise_number(m) for m in NUM.findall(text))
    idents = collections.Counter(IDENT.findall(text))
    return nums, idents


def report(label, a, b, name_a, name_b):
    only_a = (a - b)
    only_b = (b - a)
    if not only_a and not only_b:
        print(f"  {label}: identical ({sum(a.values())} occurrences)")
        return True
    for tok, n in sorted(only_a.items()):
        print(f"  {label}: {tok!r} x{n} in {name_a} but not in {name_b}")
    for tok, n in sorted(only_b.items()):
        print(f"  {label}: {tok!r} x{n} in {name_b} but not in {name_a}")
    return False


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: check_translation.py <italian.md> <english.md>")
    it, en = sys.argv[1], sys.argv[2]
    n_it, i_it = tokens(it)
    n_en, i_en = tokens(en)
    print(f"comparing {it} -> {en}")
    ok = report("numbers", n_it, n_en, it, en)
    ok &= report("identifiers", i_it, i_en, it, en)
    print("VERDICT:", "evidence preserved" if ok else "EVIDENCE DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
