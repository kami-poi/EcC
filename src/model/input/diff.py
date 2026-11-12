"""Line alignment that treats all source text, including keywords, equally."""

from difflib import SequenceMatcher
from typing import Literal

from .types import DiffDocument, DiffSpan


def make_diff(
    pre: str | None, post: str | None, *, context_lines: int = 3,
) -> DiffDocument:
    """Build a textual diff without normalizing either source.

    Lines use ``splitlines(keepends=True)``. Each hunk starts with
    ``HUNK pre_start:pre_end post_start:post_end\n`` and ends with
    ``END_HUNK\n``; all line boundaries are zero-based and half-open.
    A source record is ``KIND `` followed by the unmodified source line. If the
    source line does not end in LF, a framing LF is added outside content_range.
    Consumers use the spans rather than parsing marker-looking source text.
    Identical text (including empty/missing sides) yields NO_TEXT_CHANGE only.
    """
    if isinstance(context_lines, bool) or not isinstance(context_lines, int) or context_lines < 0:
        raise ValueError("context_lines must be a nonnegative integer")

    pre_lines = (pre if pre is not None else "").splitlines(keepends=True)
    post_lines = (post if post is not None else "").splitlines(keepends=True)
    matcher = SequenceMatcher(None, pre_lines, post_lines, autojunk=False)
    pieces: list[str] = []
    spans: list[DiffSpan] = []
    offset = 0
    hunk_count = 0

    def append_record(
        kind: Literal["ADD", "DEL", "CTX"], line: str,
        pre_range: tuple[int, int], post_range: tuple[int, int],
    ) -> None:
        nonlocal offset
        prefix = f"{kind} "
        record = prefix + line + ("" if line.endswith("\n") else "\n")
        content_start = offset + len(prefix)
        spans.append(DiffSpan(
            diff_range=(offset, offset + len(record)),
            pre_lines=pre_range,
            post_lines=post_range,
            kind=kind,
            content_range=(content_start, content_start + len(line)),
        ))
        pieces.append(record)
        offset += len(record)

    for group in matcher.get_grouped_opcodes(context_lines):
        hunk_count += 1
        header = f"HUNK {group[0][1]}:{group[-1][2]} {group[0][3]}:{group[-1][4]}\n"
        pieces.append(header)
        offset += len(header)
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    append_record("CTX", pre_lines[i], (i, i + 1), (j, j + 1))
            else:
                for i in range(i1, i2):
                    append_record("DEL", pre_lines[i], (i, i + 1), (j1, j1))
                for j in range(j1, j2):
                    append_record("ADD", post_lines[j], (i2, i2), (j, j + 1))
        footer = "END_HUNK\n"
        pieces.append(footer)
        offset += len(footer)

    if not hunk_count:
        return DiffDocument(text="NO_TEXT_CHANGE\n", spans=(), hunk_count=0)
    return DiffDocument(text="".join(pieces), spans=tuple(spans), hunk_count=hunk_count)
