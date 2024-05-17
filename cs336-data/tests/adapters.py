#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

from cs336_data.processing import extract_text_from_html_bytes, identify_language, mask_emails, mask_ip_addresses, mask_phone_numbers, detect_toxic, detect_nsfw, is_gopher_quality, detect_quality, exact_dedup, minhash_dedup


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> list[tuple[int, int]]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> list[tuple[int, int]]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> list[tuple[int, int]]:
    return mask_ip_addresses(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return detect_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return detect_toxic(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return detect_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return is_gopher_quality(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    exact_dedup(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_dedup(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
