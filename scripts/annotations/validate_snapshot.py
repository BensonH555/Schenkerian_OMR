#!/usr/bin/env python3
"""Validate a published annotation collection using Python 3.10+ stdlib only.

This verifies snapshot integrity and XML structure, not musical correctness.
Source quality warnings do not cause failure; integrity/structure errors do.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET


GROUP_PREFIX = "schenker_time_group_v1_"
STAGES = ("recognition", "assembly", "temporal")


def is_notehead(class_name: str) -> bool:
    return class_name.lower().startswith(("notehead", "gracenotehead"))


def image_size(data: bytes) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without decoding image pixels."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Only PNG and JPEG image headers are supported")
    pos = 2
    frame_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError("Invalid JPEG marker")
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        marker = data[pos]
        pos += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        length = struct.unpack(">H", data[pos:pos + 2])[0]
        if length < 2 or pos + length > len(data):
            raise ValueError("Invalid JPEG segment length")
        if marker in frame_markers:
            height, width = struct.unpack(">HH", data[pos + 3:pos + 7])
            return width, height
        if marker == 0xDA:
            break
        pos += length
    raise ValueError("JPEG frame header not found")


class Validator:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.counts = Counter()
        self.classes: set[str] = set()
        self.data_keys: set[str] = set()

    def issue(self, code: str, document: str, *, warning=False, **details):
        item = {"code": code, "document": document, **details}
        (self.warnings if warning else self.errors).append(item)

    def read_file(self, relative: str) -> bytes:
        path = self.root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("File must be a regular file inside the collection")
        return path.read_bytes()

    def parse_data(self, item: ET.Element):
        text = item.text or ""
        kind = item.get("type")
        if kind == "str":
            return text
        if kind == "int":
            return int(text)
        if kind == "float":
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("Non-finite float")
            return value
        if kind in {"list[int]", "list[float]", "list[str]"}:
            convert = {"list[int]": int, "list[float]": float,
                       "list[str]": str}[kind]
            values = [convert(value) for value in text.split()]
            if kind == "list[float]" and not all(map(math.isfinite, values)):
                raise ValueError("Non-finite float list")
            return values
        raise ValueError(f"Unsupported DataItem type: {kind}")

    def xml(self, data: bytes, document: str, canvas: tuple[int, int]):
        root = ET.fromstring(data)
        if root.tag != "Nodes":
            raise ValueError("XML root must be Nodes")
        if root.get("document") != document:
            self.issue("xml_document_mismatch", document)
        nodes = {}
        classes = set()
        groups = defaultdict(list)
        group_metadata = defaultdict(set)
        counts = Counter()
        for element in root.findall("Node"):
            node_id = int(element.findtext("Id"))
            class_element = element.find("ClassName")
            if class_element is None:
                raise ValueError(f"Missing class name on node {node_id}")
            class_name = class_element.text or ""
            if not class_name:
                self.issue("unassigned_class", document, warning=True, node=node_id)
            if node_id in nodes:
                self.issue("duplicate_node_id", document, node=node_id)
            classes.add(class_name)
            left, top, width, height = [int(element.findtext(key))
                                       for key in ("Left", "Top", "Width", "Height")]
            if width <= 0 or height <= 0:
                self.issue("nonpositive_bbox", document, node=node_id)
            elif (left < 0 or top < 0 or left + width > canvas[0]
                  or top + height > canvas[1]):
                self.issue("bbox_outside_canvas", document, warning=True,
                           node=node_id, class_name=class_name,
                           bbox=[left, top, width, height], canvas=list(canvas))
            mask = element.find("Mask")
            if mask is not None:
                counts["nodes_with_mask"] += 1
                text = (mask.text or "").strip()
                if text and text != "None":
                    if ":" in text:
                        runs = [tuple(map(int, token.split(":"))) for token in text.split()]
                        if any(len(run) != 2 for run in runs):
                            raise ValueError(f"Malformed RLE mask on node {node_id}")
                        valid = all(value in (0, 1) and count >= 0 for value, count in runs)
                        size = sum(count for _, count in runs)
                    else:
                        bits = list(map(int, text.split()))
                        valid, size = all(bit in (0, 1) for bit in bits), len(bits)
                    if valid and text == "0:0":
                        self.issue("empty_mask_placeholder", document, warning=True, node=node_id)
                    elif not valid or size != width * height:
                        self.issue("invalid_mask", document, node=node_id)
            node_data = {}
            memberships = []
            for item in element.findall("Data/DataItem"):
                key = item.get("key")
                if not key:
                    raise ValueError(f"Missing DataItem key on node {node_id}")
                if key in node_data:
                    self.issue("duplicate_data_key", document, node=node_id, key=key)
                node_data[key] = self.parse_data(item)
                self.data_keys.add(GROUP_PREFIX + "*" if key.startswith(GROUP_PREFIX) else key)
                if key in {"precedence_inlinks", "precedence_outlinks"}:
                    if item.get("type") != "list[int]":
                        self.issue("precedence_wrong_type", document, node=node_id, key=key)
                        node_data[key] = []
                if key.startswith(GROUP_PREFIX):
                    if item.get("type") != "str":
                        raise ValueError(f"Time group must be a JSON string on node {node_id}")
                    metadata = json.loads(item.text or "")
                    if not isinstance(metadata, dict) or metadata.get("version") != 1:
                        raise ValueError(f"Invalid time-group version on node {node_id}")
                    if metadata.get("kind") != "shared-time":
                        raise ValueError(f"Invalid time-group kind on node {node_id}")
                    memberships.append(key)
                    groups[key].append(node_id)
                    group_metadata[key].add(json.dumps(metadata, sort_keys=True))
                    if not is_notehead(class_name):
                        self.issue("time_group_non_notehead", document, warning=True,
                                   node=node_id, class_name=class_name, group=key)
            if len(memberships) > 1:
                self.issue("multiple_time_groups", document, node=node_id, groups=memberships)
            counts["nodes"] += 1
            counts["noteheads"] += int(is_notehead(class_name))
            counts["time_group_members"] += int(bool(memberships))
            counts["grouped_noteheads"] += int(bool(memberships) and is_notehead(class_name))
            nodes[node_id] = {
                "inlinks": [int(value) for value in (element.findtext("Inlinks") or "").split()],
                "outlinks": [int(value) for value in (element.findtext("Outlinks") or "").split()],
                "precedence_inlinks": node_data.get("precedence_inlinks", []),
                "precedence_outlinks": node_data.get("precedence_outlinks", []),
            }
        for node_id, links in nodes.items():
            for side, reverse in (("inlinks", "outlinks"), ("outlinks", "inlinks"),
                                  ("precedence_inlinks", "precedence_outlinks"),
                                  ("precedence_outlinks", "precedence_inlinks")):
                values = links[side]
                if len(values) != len(set(values)):
                    self.issue("duplicate_links", document, node=node_id, relation=side)
                for target in values:
                    if target not in nodes:
                        self.issue("dangling_link", document, node=node_id,
                                   relation=side, target=target)
                    elif node_id not in nodes[target][reverse]:
                        self.issue("nonreciprocal_link", document, node=node_id,
                                   relation=side, target=target)
            counts["syntax_edges"] += len(links["outlinks"])
            counts["precedence_edges"] += len(links["precedence_outlinks"])
        for group, metadata in group_metadata.items():
            if len(metadata) != 1:
                self.issue("inconsistent_group_metadata", document, group=group)
        counts["time_groups"] = len(groups)
        counts["observed_classes"] = len(classes)
        self.classes.update(classes)
        return counts

    def page(self, page: dict):
        document = page["document"]
        directory = page["directory"]
        contents = {}
        dimensions = {}
        for filename, expected in page["files"].items():
            relative = str(Path(directory) / filename)
            try:
                data = self.read_file(relative)
                contents[filename] = data
                if len(data) != expected["bytes"]:
                    self.issue("file_size_mismatch", document, file=relative)
                if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    self.issue("file_hash_mismatch", document, file=relative)
                if "width" in expected or "height" in expected:
                    dimensions[filename] = image_size(data)
                    if dimensions[filename] != (expected["width"], expected["height"]):
                        self.issue("image_dimensions_mismatch", document, file=relative)
                self.counts["files_checked"] += 1
            except (OSError, ValueError, KeyError, IndexError, struct.error) as exc:
                self.issue("file_read_error", document, file=relative, detail=type(exc).__name__)
        try:
            canvas = dimensions[page["annotation_canvas"]]
            original = page.get("original_image")
            if original and dimensions.get(original) != canvas:
                self.issue("original_canvas_dimensions_differ", document, warning=True)
            counts = self.xml(contents["mung.xml"], document, canvas)
            for key, value in page.get("counts", {}).items():
                if counts.get(key, 0) != value:
                    self.issue("page_count_mismatch", document, field=key,
                               expected=value, actual=counts.get(key, 0))
            self.counts.update({key: value for key, value in counts.items()
                                if key != "observed_classes"})
            workflow = json.loads(contents["workflow.json"])
            for key in ("document", "stages", "status_origin", "all_three_explicitly_done"):
                if key not in workflow:
                    self.issue("workflow_missing_field", document, field=key)
                elif workflow[key] != page[key]:
                    self.issue("workflow_manifest_mismatch", document, field=key)
            fully_done = all(page["stages"].get(stage) == "done" and
                             page["status_origin"].get(stage) == "explicit_stage"
                             for stage in STAGES)
            if fully_done != page["all_three_explicitly_done"]:
                self.issue("completion_flag_mismatch", document)
            self.counts["all_three_explicitly_done"] += int(fully_done)
        except (ET.ParseError, ValueError, TypeError, KeyError) as exc:
            self.issue("malformed_page", document, detail=str(exc))
        self.counts["pages"] += 1

    def run(self):
        manifest = json.loads(self.read_file("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema version")
        seen_documents, seen_directories = set(), set()
        for page in manifest["pages"]:
            for key, seen in (("document", seen_documents), ("directory", seen_directories)):
                if page[key] in seen:
                    self.issue("duplicate_manifest_page", page["document"], field=key)
                seen.add(page[key])
            self.page(page)
        self.counts["observed_classes"] = len(self.classes)
        self.counts["data_key_families"] = len(self.data_keys)
        for key, expected in manifest.get("counts", {}).items():
            if self.counts.get(key, 0) != expected:
                self.issue("collection_count_mismatch", "collection", field=key,
                           expected=expected, actual=self.counts.get(key, 0))
        return {"valid": not self.errors, "counts": dict(sorted(self.counts.items())),
                "errors": self.errors, "warnings": self.warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path, help="Directory containing manifest.json")
    parser.add_argument("--report", type=Path, help="Also write the JSON result to this file")
    args = parser.parse_args()
    try:
        report = Validator(args.collection).run()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"valid": False, "counts": {}, "warnings": [],
                  "errors": [{"code": "invalid_manifest", "detail": str(exc)}]}
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
