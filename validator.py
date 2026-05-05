"""Vietnamese license plate validation and normalization."""

import re

# Vietnamese license plate patterns (after normalization)
PATTERNS = [
    r"^\d{2}[A-Z]\d{5,6}$",          # e.g. 90B245230 (new format)
    r"^\d{2}[A-Z]\d{4,5}$",          # e.g. 51A12345
    r"^\d{2}[A-Z]{1,2}\d{4,5}$",    # e.g. 51AB12345
]

# Common OCR misreads: letter -> digit mapping
OCR_FIXES = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8"}


def is_valid_plate(text: str) -> bool:
    """Check whether text matches any known Vietnamese plate pattern."""
    cleaned = text.replace("-", "").replace(" ", "").replace(".", "").upper()
    return any(re.match(p, cleaned) for p in PATTERNS)


def normalize(text: str) -> str:
    """Normalize plate text: uppercase, strip spaces/dashes, fix common OCR errors.

    Digits at positions 0,1,3,4,5,6 that were misread as letters are corrected.
    """
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    result = []
    for i, ch in enumerate(text):
        # Positions that should be digits in standard plates
        if i in (0, 1, 3, 4, 5, 6) and ch in OCR_FIXES:
            result.append(OCR_FIXES[ch])
        else:
            result.append(ch)
    return "".join(result)
