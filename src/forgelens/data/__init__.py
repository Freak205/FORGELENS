"""Dataset contracts, safe fixtures, and leakage-safe split helpers."""

from forgelens.data.aiforge import AIForgeDocForgeryDataset
from forgelens.data.cord import CordAuthenticDataset
from forgelens.data.fixtures import FictionalDocumentFixtures
from forgelens.data.splits import GroupedSplit, grouped_split
from forgelens.data.types import DocumentSample

__all__ = [
    "AIForgeDocForgeryDataset",
    "CordAuthenticDataset",
    "DocumentSample",
    "FictionalDocumentFixtures",
    "GroupedSplit",
    "grouped_split",
]
