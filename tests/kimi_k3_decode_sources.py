"""Where the decode kernel's source contracts read their sources from.

A source contract is a claim about the text of the kernel, so it reads that
text. Two of the decode headers grew past a thousand lines and were decomposed
into a directory of focused parts under a thin umbrella that includes them in
dependency order. That is a change in how the text is stored, not in what it
says, and a contract that broke on it would be asserting the storage.

So a header is read here as the header it is: an umbrella resolves to its own
lines with each of its parts inlined where the include names it, in the order
the umbrella includes them. Only includes that name a subdirectory of this
source root are followed, which is exactly the umbrella-to-part edge -- a part
including its predecessor by bare name is left alone, so no part is inlined
twice, and a header including a sibling header is left alone, so the closure a
contract reads stays the one file it asked for.

``headers`` is the other half: a sweep that used to glob one directory has to
see the parts too, and it has to name them by a path rather than a base name,
because ``dependency_local/kernel.cuh`` and ``kernel.cuh`` are two files.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "csrc" / "kimi_k3_decode"

#: An umbrella's include of one of its own parts. Anchored to a whole line so a
#: mention inside a comment or a string cannot be inlined, and resolved against
#: this directory so an include of a bundled third-party path is left alone.
_PART_INCLUDE = re.compile(
    r'^#include "([A-Za-z0-9_]+/[A-Za-z0-9_]+\.cuh)"$', re.MULTILINE
)


def _inline(part: re.Match[str]) -> str:
    relative = part.group(1)
    if not (SOURCE_ROOT / relative).is_file():
        return part.group(0)
    return read(relative)


def read(relative: str) -> str:
    """One decode header's source, with the parts it umbrellas inlined."""
    text = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
    return _PART_INCLUDE.sub(_inline, text)


def headers() -> tuple[Path, ...]:
    """Every decode header, umbrellas and parts alike, in a stable order."""
    return tuple(sorted(SOURCE_ROOT.rglob("*.cuh")))


def includable() -> tuple[Path, ...]:
    """The headers a caller includes, which is the top level of the directory.

    A part is reached only through its umbrella, and ``read`` on an umbrella
    already carries every part's text, so a sweep over these covers the whole
    directory exactly once.
    """
    return tuple(sorted(SOURCE_ROOT.glob("*.cuh")))


def name(path: Path) -> str:
    """A header's path relative to the source root, as ``read`` takes it."""
    return path.relative_to(SOURCE_ROOT).as_posix()


def parts_of(relative: str) -> tuple[str, ...]:
    """The parts one umbrella includes, in the order it includes them."""
    text = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
    return tuple(
        part for part in _PART_INCLUDE.findall(text)
        if (SOURCE_ROOT / part).is_file()
    )
