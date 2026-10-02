"""Highlight, in yellow, the numbers in an answer that the language model wrote itself rather than took from a tool.

After the assistant answers, every number in its reply is compared with every number in the tool results
of the conversation so far. A number counts as matching when it equals a tool value at the precision it is
written (7.43 matches 7.4318; 85% matches a similarity of 0.85). Then:

- plain:  the number matches a tool value; it was looked up or computed by code
- yellow: it matches nothing; the model produced it itself (its own arithmetic, rounding or recollection)
- none:   numbers the person typed in their question, small counts (below 10, written without decimals,
          such as "2 pairs"), conventions (the 10 in "10^0.31", the 95% in "95% CI"), and anything inside an
          identifier or code (CHEMBL600647, `pki_range_ge_1.0`)

An unhighlighted number means "this value appears in a tool result", not "this sentence uses it correctly"; a
model could quote a real number in the wrong place. Only yellow is drawn: marking tool numbers too (green, until
2026-10-02) made answers busy, and the person needs to see only the numbers to check. Sentences the model itself labels as general knowledge are tinted yellow
too, since the system prompt requires that label for claims not taken from the data.
"""
import json
import re

YELLOW = "rgba(234, 179, 8, 0.30)"
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")

# A number standing on its own: not part of a word or identifier (CHEMBL600647, C24H36O3, fold_1). It may use
# a Unicode minus, thousands separators, decimals, scientific notation written as "1.02×10⁻⁵", and a "%".
NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[−-])?(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*[×x]\s*10(?P<exponent>[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?(?P<percent>\s?%)?(?!(?!x\b)\w)")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def tool_values(results):
    """Every number in a list of tool results (JSON text): numeric values anywhere in the structure.

    Strings are not searched, so digits inside a SMILES or an assay description are not counted.
    """
    values = []

    def walk(node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            values.append(float(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for text in results:
        try:
            walk(json.loads(text))
        except (TypeError, ValueError):
            continue
    return values


def parse(match):
    """(value, decimals shown) for a regex match."""
    digits = match.group("digits").replace(",", "")
    value = float(digits)
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    if match.group("exponent"):
        exponent = int(match.group("exponent").translate(SUPERSCRIPTS))
        value *= 10 ** exponent
        decimals -= exponent
    if match.group("sign"):
        value = -value
    return value, decimals


def matches(value, decimals, percent, values):
    """Does this written number equal some tool value, at the precision it is written?"""
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    candidates = [v * 100 for v in values] + values if percent else values
    # A sign can be dropped in prose ("a drop of 2.42" for a difference of -2.42).
    return any(abs(abs(value) - abs(v)) <= tolerance for v in candidates)


def classify(text, values, given=()):
    """[(start, end, kind)] for each number in plain text: kind is "tool", "model" or None (not highlighted)."""
    given_values = [v for v, _ in (parse(m) for m in NUMBER.finditer(" ".join(given)))]
    found = []
    for match in NUMBER.finditer(text):
        value, decimals = parse(match)
        small_count = decimals <= 0 and abs(value) < 10 and not match.group("percent") and not match.group("exponent")
        # A numbered-list marker ("10. " at the start of a line) is not a number in the answer.
        line_start = text.rfind("\n", 0, match.start()) + 1
        list_marker = not text[line_start:match.start()].strip() and text[match.end():match.end() + 2] in (". ", ".\n")
        # Conventions, not results: the base of a power ("10^0.31") and a confidence level ("95% CI").
        after = text[match.end():match.end() + 15].lower()
        convention = after.startswith("^") or (match.group("percent") and value in (90, 95, 99)
                                               and re.match(r"\W*(ci\b|conf|interval|level)", after))
        if small_count or list_marker or convention or any(abs(value - g) <= 0.5 * 10 ** -max(decimals, 0) for g in given_values):
            kind = None
        else:
            kind = "tool" if matches(value, decimals, bool(match.group("percent")), values) else "model"
        found.append((match.start(), match.end(), kind))
    return found


# A sentence: runs to ".", "!" or "?", but a dot between two digits (7.43) is a decimal point, not an end.
SENTENCE = re.compile(r"(?:[^.!?\n]|(?<=\d)\.(?=\d))+[.!?]?")


def general_knowledge_sentences(text):
    """Character spans of sentences that the model labels as general knowledge."""
    return [(s.start(), s.end()) for s in SENTENCE.finditer(text) if "general knowledge" in s.group(0).lower()]


def highlight(answer, results, given=()):
    """The answer as Markdown with HTML highlights, and a count of each kind of number.

    Code spans and blocks are left untouched. "<" in the model's own text is escaped first, so the only HTML
    in the result is the highlighting added here.
    """
    values = tool_values(results)
    counts = {"tool": 0, "model": 0}
    pieces, position = [], 0
    for code in CODE.finditer(answer):
        pieces.append(("text", answer[position:code.start()]))
        pieces.append(("code", code.group(0)))
        position = code.end()
    pieces.append(("text", answer[position:]))

    output = []
    for kind, piece in pieces:
        if kind == "code":
            output.append(piece)
            continue
        piece = piece.replace("<", "&lt;")
        # Every highlight is an opening tag and a closing tag at positions in the original text. They are
        # inserted together, closings before openings at the same position, so tags always nest.
        tags = []
        for start, end, number_kind in classify(piece, values, given):
            if number_kind:
                counts[number_kind] += 1
            if number_kind == "model":
                tags += [(start, 3, f'<span style="background-color:{YELLOW};border-radius:3px;padding:0 2px">'),
                         (end, 0, "</span>")]
        for start, end in general_knowledge_sentences(piece):
            tags += [(start, 2, f'<span style="background-color:{YELLOW};border-radius:3px">'), (end, 1, "</span>")]
        result, position = [], 0
        for at, _, tag in sorted(tags):
            result += [piece[position:at], tag]
            position = at
        result.append(piece[position:])
        output.append("".join(result))
    return "".join(output), counts
