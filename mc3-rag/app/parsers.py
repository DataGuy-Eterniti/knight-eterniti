"""File parsers. Pure CPU, no torch import, so they can run in a spawned worker
process that the server can kill if a file hangs.

parse_file(path) -> {"kind": str, "units": [{"text": str, "loc": str, "row": bool}],
                     "withdrawn": bool}
Raises Skip(reason) for anything that must not be indexed (encrypted, unknown type,
unreadable, binary).
"""
import csv
import io
import os
import re

TEXT_EXT = {".txt", ".log", ".py", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg",
            ".toml", ".sh", ".xml", ".html", ".htm", ".rst", ".conf", ".sql", ".js",
            ".ts", ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rs", ".env.example"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
XLSX_EXT = {".xlsx", ".xlsm"}
CSV_EXT = {".csv", ".tsv"}

MAX_BYTES = 200 * 1024 * 1024
MAX_ROWS_PER_SHEET = 20000
WITHDRAWN_NAME = re.compile(r"withdrawn|superseded|obsolete|deprecated|retired", re.I)
# Only phrases that say THIS document is withdrawn. A current revision that mentions
# "revision 1 was withdrawn" must not be flagged.
WITHDRAWN_TEXT = re.compile(
    r"\bsuperseded\s+by\b"
    r"|\bthis\s+(?:revision|document|datasheet|version|edition|spec(?:ification)?)\s+"
    r"(?:is|has\s+been)\s+(?:now\s+)?(?:withdrawn|superseded|obsolete|retired)\b"
    r"|\bstatus\s*[:=]\s*(?:withdrawn|superseded|obsolete|retired)\b"
    r"|\bdo\s+not\s+use\s+for\s+new\s+designs\b", re.I)


class Skip(Exception):
    pass


def kind_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXT:
        return "pdf"
    if ext in DOCX_EXT:
        return "docx"
    if ext in XLSX_EXT:
        return "xlsx"
    if ext in CSV_EXT:
        return "csv"
    if ext in IMAGE_EXT:
        return "image"
    if ext in TEXT_EXT:
        return "text"
    return None


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    if s.endswith(".0") and re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return re.sub(r"\s+", " ", s)


def _rows_to_units(rows, where, max_rows):
    """rows: list of lists. First row with >=2 non-empty cells is the header."""
    units = []
    header = None
    count = 0
    for i, row in enumerate(rows):
        cells = [_clean(c) for c in row]
        if not any(cells):
            continue
        if header is None and sum(1 for c in cells if c) >= 2:
            header = cells
            units.append({"text": f"{where} columns: " + " | ".join(c for c in cells if c),
                          "loc": f"{where} header", "row": True})
            continue
        if header:
            parts = []
            for j, c in enumerate(cells):
                if not c:
                    continue
                h = header[j] if j < len(header) and header[j] else f"col{j + 1}"
                parts.append(f"{h}: {c}")
            text = " | ".join(parts)
        else:
            text = " | ".join(c for c in cells if c)
        units.append({"text": f"{where} | {text}", "loc": f"{where} row {i + 1}", "row": True})
        count += 1
        if count >= max_rows:
            break
    return units


def parse_pdf(path):
    from pypdf import PdfReader
    try:
        reader = PdfReader(path, strict=False)
    except Exception as e:  # noqa: BLE001
        raise Skip(f"pdf open failed: {e}")
    if reader.is_encrypted:
        # A file that needs a password is never a source. Don't try to open it.
        raise Skip("encrypted pdf")
    units = []
    for n, page in enumerate(reader.pages, 1):
        text = ""
        try:
            text = page.extract_text(extraction_mode="layout") or ""
            # Layout mode pads table columns with spaces; turn wide gaps into separators.
            text = "\n".join(re.sub(r" {3,}", " | ", ln).rstrip() for ln in text.splitlines())
        except Exception:  # noqa: BLE001
            text = ""
        if len(text.strip()) < 20:
            try:
                text = page.extract_text() or text
            except Exception:  # noqa: BLE001
                pass
        if text.strip():
            units.append({"text": text, "loc": f"page {n}", "row": False})
    if not units:
        # Scanned PDF: let the server render and OCR it with the vision model.
        return {"kind": "pdf_scanned", "units": [], "pages": len(reader.pages)}
    return {"kind": "pdf", "units": units}


def parse_docx(path):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    try:
        d = docx.Document(path)
    except Exception as e:  # noqa: BLE001  (encrypted .docx is an OLE file and fails here)
        raise Skip(f"docx open failed: {e}")
    units = []
    buf = []
    heading = ""

    def flush():
        if buf:
            units.append({"text": "\n".join(buf), "loc": heading or "body", "row": False})
            buf.clear()

    body = d.element.body
    tcount = 0
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            t = p.text.strip()
            if not t:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                flush()
                heading = t
            buf.append(t)
            if sum(len(x) for x in buf) > 1200:
                flush()
        elif tag == "tbl":
            flush()
            tcount += 1
            table = Table(child, d)
            rows = []
            for r in table.rows:
                cells = []
                seen = set()
                for c in r.cells:  # merged cells repeat; keep each once
                    if id(c._tc) in seen:
                        continue
                    seen.add(id(c._tc))
                    cells.append(c.text)
                rows.append(cells)
            where = f"table {tcount}" + (f" ({heading})" if heading else "")
            units.extend(_rows_to_units(rows, where, MAX_ROWS_PER_SHEET))
    flush()
    for s in d.sections:  # headers/footers sometimes carry doc number / revision
        for part in (s.header, s.footer):
            try:
                t = "\n".join(p.text for p in part.paragraphs if p.text.strip())
            except Exception:  # noqa: BLE001
                t = ""
            if t:
                units.append({"text": t, "loc": "header/footer", "row": False})
    return {"kind": "docx", "units": units}


def parse_xlsx(path):
    import openpyxl
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001
        raise Skip(f"xlsx open failed: {e}")
    units = []
    try:
        for ws in wb.worksheets:
            rows = []
            for r in ws.iter_rows(values_only=True):
                rows.append(list(r))
                if len(rows) > MAX_ROWS_PER_SHEET + 50:
                    break
            units.extend(_rows_to_units(rows, f"sheet {ws.title}", MAX_ROWS_PER_SHEET))
    finally:
        wb.close()
    return {"kind": "xlsx", "units": units}


def _read_text(path):
    with open(path, "rb") as f:
        data = f.read(MAX_BYTES)
    if b"\x00" in data[:8192]:
        raise Skip("binary content")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_csv(path):
    text = _read_text(path)
    delim = "\t" if path.lower().endswith(".tsv") else ","
    try:
        delim = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except Exception:  # noqa: BLE001
        pass
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    return {"kind": "csv", "units": _rows_to_units(rows, "csv", MAX_ROWS_PER_SHEET)}


def parse_text(path):
    text = _read_text(path)
    return {"kind": "text", "units": [{"text": text, "loc": "file", "row": False}]}


def parse_file(path):
    if not os.path.isfile(path):
        raise Skip("not a regular file")
    if os.path.getsize(path) > MAX_BYTES:
        raise Skip("too large")
    kind = kind_of(path)
    if kind is None:
        raise Skip("unknown type")
    # Fail fast and cleanly on files we may not read (chmod 000, DAC_OVERRIDE dropped).
    with open(path, "rb") as f:
        head = f.read(8)
    if kind == "image":
        return {"kind": "image", "units": []}
    if kind in ("docx", "xlsx") and head[:4] != b"PK\x03\x04":
        raise Skip("not a zip container (likely encrypted Office file)")
    fn = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx,
          "csv": parse_csv, "text": parse_text}[kind]
    out = fn(path)
    first = " ".join(u["text"] for u in out.get("units", [])[:3])[:3000]
    out["withdrawn"] = bool(WITHDRAWN_NAME.search(os.path.basename(path))
                            or WITHDRAWN_TEXT.search(first))
    return out


def serve():
    """Line-delimited JSON loop on stdin/stdout. server.py runs this in a separate plain
    Python process, so a file that hangs or crashes the parser only costs that file."""
    import json
    import sys
    out = sys.stdout
    sys.stdout = sys.stderr  # nothing a library prints may corrupt the reply channel
    for line in sys.stdin:
        try:
            res = parse_file(json.loads(line)["path"])
        except Skip as e:
            res = {"skip": str(e)}
        except PermissionError:
            res = {"skip": "permission denied"}
        except Exception as e:  # noqa: BLE001
            res = {"skip": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res) + "\n")
        out.flush()
