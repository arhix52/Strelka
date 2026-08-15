#!/usr/bin/env python3
"""Build an xctrace GPU-counter template and summarize its XML export."""

from __future__ import annotations

import argparse
import collections
import math
import plistlib
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class XmlValue:
    raw: str
    formatted: str


def _archive_object(objects: list[object], reference: object, context: str) -> object:
    if not isinstance(reference, plistlib.UID):
        raise ValueError(f"{context} is not an NSKeyedArchive UID")
    try:
        return objects[reference.data]
    except IndexError as error:
        raise ValueError(f"{context} points outside the NSKeyedArchive") from error


def _recording_state(archive: dict[str, object]) -> tuple[list[object], dict[str, object]]:
    try:
        objects = archive["$objects"]
        top = archive["$top"]
    except KeyError as error:
        raise ValueError("input is not an Instruments NSKeyedArchive template") from error
    if not isinstance(objects, list) or not isinstance(top, dict):
        raise ValueError("malformed Instruments template")

    control_reference = top.get("com.apple.xray.recordingControlState")
    if control_reference is None:
        command = _archive_object(objects, top.get("templateRunCommand"), "templateRunCommand")
        if not isinstance(command, dict):
            raise ValueError("malformed Instruments template command")
        control_reference = command.get("_recordingControlState")

    control = _archive_object(objects, control_reference, "recording control state")
    if not isinstance(control, dict):
        raise ValueError("malformed Instruments recording control state")
    state = _archive_object(objects, control.get("state"), "recording state")
    if not isinstance(state, dict) or not isinstance(state.get("NS.keys"), list) or not isinstance(
        state.get("NS.objects"), list
    ):
        raise ValueError("malformed Instruments recording-options dictionary")
    return objects, state


def _decoded_key(objects: list[object], reference: object) -> object:
    value = _archive_object(objects, reference, "recording-options key")
    if isinstance(value, dict) and "NS.string" in value:
        return value["NS.string"]
    return value


def _set_keyed_value(objects: list[object], state: dict[str, object], name: str, value: object) -> None:
    keys = state["NS.keys"]
    values = state["NS.objects"]
    assert isinstance(keys, list) and isinstance(values, list)

    objects.append(value)
    value_reference = plistlib.UID(len(objects) - 1)
    for index, key_reference in enumerate(keys):
        if _decoded_key(objects, key_reference) == name:
            values[index] = value_reference
            return

    objects.append(name)
    keys.append(plistlib.UID(len(objects) - 1))
    values.append(value_reference)


def make_template(source: Path, destination: Path, profile_id: int) -> None:
    with source.open("rb") as stream:
        archive = plistlib.load(stream)
    if not isinstance(archive, dict):
        raise ValueError("malformed Instruments template")

    objects, state = _recording_state(archive)
    _set_keyed_value(objects, state, "counterscounterprofile", profile_id)
    _set_keyed_value(objects, state, "countersshaderprofiler", False)

    with destination.open("wb") as stream:
        plistlib.dump(archive, stream, fmt=plistlib.FMT_BINARY, sort_keys=False)


def _xml_value(element: ET.Element, references: dict[str, XmlValue]) -> XmlValue | None:
    reference = element.get("ref")
    if reference is not None:
        return references.get(reference)
    raw = (element.text or "").strip()
    return XmlValue(raw=raw, formatted=element.get("fmt", raw))


def summarize(stream: BinaryIO) -> int:
    # xctrace assigns an id to values and emits <tag ref="id"/> when a value
    # repeats. Keep only the field types used below; start times and sample
    # indices account for most unique ids and never need to be retained.
    references: dict[str, XmlValue] = {}
    totals: dict[str, list[float | int]] = collections.defaultdict(lambda: [0.0, 0])
    row_count = 0

    for _, element in ET.iterparse(stream, events=("end",)):
        if (
            element.tag in {"duration", "gpu-counter-name", "fixed-decimal"}
            and element.get("id") is not None
        ):
            value = _xml_value(element, references)
            if value is not None:
                references[element.get("id", "")] = value

        if element.tag != "row":
            continue

        row_count += 1
        fields: dict[str, XmlValue] = {}
        for child in element:
            if child.tag not in {"duration", "gpu-counter-name", "fixed-decimal"}:
                continue
            value = _xml_value(child, references)
            if value is not None:
                fields[child.tag] = value

        name = fields.get("gpu-counter-name")
        counter_value = fields.get("fixed-decimal")
        if name is not None and counter_value is not None:
            try:
                numeric_value = float(counter_value.raw)
                duration = int(fields.get("duration", XmlValue("1", "1")).raw)
            except ValueError:
                element.clear()
                continue
            if math.isfinite(numeric_value):
                weight = max(duration, 1)
                total = totals[name.formatted]
                total[0] = float(total[0]) + numeric_value * weight
                total[1] = int(total[1]) + weight
        element.clear()

    if not totals:
        print(
            f"no GPU hardware counter values found ({row_count} exported rows); "
            "check the Instruments Counter Set/profile",
            file=sys.stderr,
        )
        return 2

    for name in sorted(totals):
        weighted_sum, total_duration = totals[name]
        average = float(weighted_sum) / max(int(total_duration), 1)
        print(f"{name[:52]:<52} {average:10.2f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("make-template")
    template_parser.add_argument("source", type=Path)
    template_parser.add_argument("destination", type=Path)
    template_parser.add_argument("--profile-id", type=int, required=True)

    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if arguments.command == "make-template":
            make_template(arguments.source, arguments.destination, arguments.profile_id)
            return 0
        return summarize(sys.stdin.buffer)
    except (OSError, ValueError, plistlib.InvalidFileException, ET.ParseError) as error:
        print(f"gpu counter processing failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
