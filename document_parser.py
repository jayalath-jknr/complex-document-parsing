"""
Complex Document Parser using Docling
Supports: PDF, DOCX, PPTX, HTML, Images, Markdown, CSV, AsciiDoc
Exports:  Markdown, JSON, YAML, plain text, DocTags, tables (CSV/HTML)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Converter factory
# ---------------------------------------------------------------------------

def build_converter(
    ocr: bool = True,
    ocr_lang: list[str] | None = None,
    do_table_structure: bool = True,
    do_cell_matching: bool = True,
    num_threads: int = 4,
    device: str = "auto",
) -> DocumentConverter:
    """
    Build a DocumentConverter that handles every supported format and applies
    fine-grained pipeline options for PDF and DOCX.

    Parameters
    ----------
    ocr              : enable EasyOCR for scanned pages
    ocr_lang         : list of ISO 639-1 language codes, e.g. ["en", "de"]
    do_table_structure: detect table structure
    do_cell_matching : match cell content to detected table cells (higher accuracy)
    num_threads      : CPU threads for the accelerator
    device           : "auto" | "cpu" | "cuda" | "mps"
    """
    ocr_lang = ocr_lang or ["en"]

    accelerator_map = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }
    accel_device = accelerator_map.get(device.lower(), AcceleratorDevice.AUTO)

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = ocr
    pdf_options.do_table_structure = do_table_structure
    pdf_options.table_structure_options = TableStructureOptions(
        do_cell_matching=do_cell_matching
    )
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=accel_device,
    )
    if ocr:
        pdf_options.ocr_options.lang = ocr_lang

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pdf_options,
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_document(conv_result, output_dir: Path) -> dict[str, Path]:
    """
    Write all supported export formats for a single conversion result.

    Returns a dict mapping format name -> output path.
    """
    stem = conv_result.input.file.stem
    doc = conv_result.document
    paths: dict[str, Path] = {}

    # Markdown
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    paths["markdown"] = md_path

    # Plain text (strict markdown, no formatting tokens)
    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text(doc.export_to_markdown(strict_text=True), encoding="utf-8")
    paths["text"] = txt_path

    # JSON
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["json"] = json_path

    # YAML
    yaml_path = output_dir / f"{stem}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(doc.export_to_dict(), allow_unicode=True),
        encoding="utf-8",
    )
    paths["yaml"] = yaml_path

    # DocTags (Docling native structured format)
    doctags_path = output_dir / f"{stem}.doctags"
    doctags_path.write_text(doc.export_to_doctags(), encoding="utf-8")
    paths["doctags"] = doctags_path

    return paths


def export_tables(conv_result, output_dir: Path) -> list[dict]:
    """
    Extract every table detected in the document and export as CSV and HTML.

    Returns a list of dicts with metadata about each table.
    """
    try:
        import pandas as pd  # optional but highly recommended
    except ImportError:
        _log.warning("pandas not installed – skipping table export (pip install pandas)")
        return []

    stem = conv_result.input.file.stem
    doc = conv_result.document
    table_meta = []

    for idx, table in enumerate(doc.tables, start=1):
        df: "pd.DataFrame" = table.export_to_dataframe(doc=doc)

        csv_path = output_dir / "tables" / f"{stem}-table-{idx}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        html_path = output_dir / "tables" / f"{stem}-table-{idx}.html"
        html_path.write_text(table.export_to_html(doc=doc), encoding="utf-8")

        table_meta.append(
            {
                "index": idx,
                "rows": len(df),
                "cols": len(df.columns),
                "csv": str(csv_path),
                "html": str(html_path),
            }
        )
        _log.info("  Table %d: %d rows × %d cols → %s", idx, len(df), len(df.columns), csv_path)

    return table_meta


# ---------------------------------------------------------------------------
# Parse summary
# ---------------------------------------------------------------------------

def summarise(conv_result) -> dict:
    """Return a lightweight summary dict for a conversion result."""
    doc = conv_result.document

    # Count elements by type label
    element_counts: dict[str, int] = {}
    for item, _ in doc.iterate_items():
        label = type(item).__name__
        element_counts[label] = element_counts.get(label, 0) + 1

    # num_pages may be a callable or an int depending on docling version
    num_pages = getattr(doc, "num_pages", None)
    if callable(num_pages):
        num_pages = num_pages()

    return {
        "file": conv_result.input.file.name,
        "format": str(conv_result.input.format),
        "pages": num_pages,
        "tables": len(doc.tables),
        "elements": element_counts,
    }


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_documents(
    input_paths: list[Path],
    output_dir: Path,
    ocr: bool = True,
    ocr_lang: list[str] | None = None,
    do_table_structure: bool = True,
    num_threads: int = 4,
    device: str = "auto",
) -> list[dict]:
    """
    Convert all documents in *input_paths* and write exports to *output_dir*.

    Returns a list of per-document result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = build_converter(
        ocr=ocr,
        ocr_lang=ocr_lang,
        do_table_structure=do_table_structure,
        num_threads=num_threads,
        device=device,
    )

    results = []
    t_start = time.time()

    for res in converter.convert_all(input_paths):
        t0 = time.time()

        summary = summarise(res)
        file_paths = export_document(res, output_dir)
        table_meta = export_tables(res, output_dir)

        elapsed = time.time() - t0
        summary["elapsed_s"] = round(elapsed, 3)
        summary["tables_exported"] = table_meta
        summary["outputs"] = {k: str(v) for k, v in file_paths.items()}

        results.append(summary)

        _log.info(
            "[%s] converted in %.2fs | %d table(s) | outputs → %s",
            res.input.file.name,
            elapsed,
            len(table_meta),
            output_dir,
        )

    total = time.time() - t_start
    _log.info("Finished %d document(s) in %.2fs total.", len(results), total)

    # Write a combined manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"total_elapsed_s": round(total, 3), "documents": results}, indent=2),
        encoding="utf-8",
    )
    _log.info("Manifest written to %s", manifest_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document_parser",
        description="Complex multi-format document parser powered by Docling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE",
        help="One or more document paths (PDF, DOCX, PPTX, HTML, image, MD, CSV, AsciiDoc).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="scratch",
        metavar="DIR",
        help="Directory to write exported files into.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR (useful for native-text PDFs).",
    )
    parser.add_argument(
        "--ocr-lang",
        nargs="+",
        default=["en"],
        metavar="LANG",
        help="OCR language codes, e.g. --ocr-lang en de fr",
    )
    parser.add_argument(
        "--no-table-structure",
        action="store_true",
        help="Skip table structure detection.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        metavar="N",
        help="Number of CPU threads for the accelerator.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    cli = build_cli()
    args = cli.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    input_paths = [Path(p) for p in args.inputs]
    missing = [p for p in input_paths if not p.exists()]
    if missing:
        _log.error("The following inputs were not found:\n  %s", "\n  ".join(str(p) for p in missing))
        sys.exit(1)

    parse_documents(
        input_paths=input_paths,
        output_dir=Path(args.output_dir),
        ocr=not args.no_ocr,
        ocr_lang=args.ocr_lang,
        do_table_structure=not args.no_table_structure,
        num_threads=args.threads,
        device=args.device,
    )


if __name__ == "__main__":
    main()
