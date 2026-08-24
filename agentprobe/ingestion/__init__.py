"""Ingestion and processing: turn raw documents into clean, tagged sections."""

from .loaders import load_document, load_folder
from .cleaner import clean_text
from .chunker import chunk_document
from .tagger import tag_chunks

__all__ = ["load_document", "load_folder", "clean_text", "chunk_document", "tag_chunks"]
