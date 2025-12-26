import json
import logging
import os
from pathlib import Path
from typing import Iterable

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader

logger = logging.getLogger(__name__)

# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = ["key", "PDF", "pdf", "docx", "xlsx", "pptx", "csv", "tsv", "txt"]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FORMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)


def segment_blocks(nb_content: dict) -> tuple[list[str], list[str]]:
    """Extract cell sources and types from a notebook JSON structure."""
    cells: list[str] = []
    cell_types: list[str] = []

    for cell in nb_content.get("cells", []):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])
        text = "".join(source)
        cells.append(text)
        cell_types.append(cell_type)

    return cells, cell_types


class PersonalCopilotDatasetReader(BaseDiskReader):
    """Reads raw Hugging Face org repos from disk into Documents."""

    name = "👩🏽‍💻 Personal Copilot Dataset Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        skip: int = 0,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder=data_folder,
            limit=limit,
            skip=skip,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )

    def _should_skip(self, filepath: str) -> bool:
        ext = Path(filepath).suffix.lower().lstrip(".")
        return ext in ANTI_FORMATS

    def _read_notebook(self, filepath: str) -> Iterable[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            cells, _ = segment_blocks(content)
            for cell in cells:
                if cell.strip():
                    yield cell
        except Exception as exc:  # best-effort; skip unreadable notebooks
            logger.warning("Failed to read notebook %s: %s", filepath, exc)

    def read_file(self, filepath: str):
        """Yield Documents from a single file path."""
        # Skip blocked formats early.
        if self._should_skip(filepath):
            return

        if filepath.endswith(".ipynb"):
            for idx, cell_text in enumerate(self._read_notebook(filepath)):
                doc = self.get_document_from_dict(
                    {"id": f"{filepath}-{idx}", "text": cell_text},
                    filepath,
                    idx,
                )
                if doc:
                    yield doc
            return

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as exc:
            logger.warning("Failed to read file %s: %s", filepath, exc)
            return

        if not text or not text.strip():
            return

        doc = self.get_document_from_dict(
            {"id": filepath, "text": text},
            filepath,
            0,
        )
        if doc:
            yield doc