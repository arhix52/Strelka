"""Minimal OpenEXR scanline reader -- no Blender, no OpenEXR bindings, no pip.

Why this exists
---------------
`compare.py` used to read EXRs through `bpy.data.images.load`, which meant the
whole comparison ladder could only be graded under Blender. That is fine on the
machine that *produces* the Cycles references, and wrong everywhere else: the
references are already checked in, so grading a render needs a reader, not a
renderer. It also made the OptiX backend -- which only exists on Linux/Windows,
where nobody wants a 400 MB Blender install to diff two images -- ungradeable.

Scope is deliberately the subset this repo writes and reads, and nothing else:

  * scanline images (not tiled, not deep, not multipart)
  * ZIP / ZIPS / NONE / RLE compression
  * HALF and FLOAT channels

Anything outside that raises, loudly, rather than returning a plausible-looking
array. A silently mis-decoded reference is worse than no reader at all -- it
would show up as a shading regression and be chased in the renderer.

PIZ, PXR24, B44 and DWA are not implemented. Blender writes ZIP by default and
tinyexr writes ZIP, so nothing in the ladder needs them; if that changes, this
raises `ExrError` naming the compression instead of guessing.
"""

import struct
import zlib

import numpy as np

EXR_MAGIC = 0x01312F76

# Values of the `compression` header attribute.
COMPRESSION_NONE = 0
COMPRESSION_RLE = 1
COMPRESSION_ZIPS = 2
COMPRESSION_ZIP = 3

# Scanlines per compressed chunk, indexed by compression id.
_LINES_PER_BLOCK = {
    COMPRESSION_NONE: 1,
    COMPRESSION_RLE: 1,
    COMPRESSION_ZIPS: 1,
    COMPRESSION_ZIP: 16,
}

_COMPRESSION_NAMES = {
    0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
    5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB",
}

# Values of the pixel-type field of a channel.
PIXEL_UINT = 0
PIXEL_HALF = 1
PIXEL_FLOAT = 2

_PIXEL_DTYPE = {
    PIXEL_UINT: np.dtype("<u4"),
    PIXEL_HALF: np.dtype("<f2"),
    PIXEL_FLOAT: np.dtype("<f4"),
}


class ExrError(Exception):
    pass


def _read_null_terminated(buf, offset):
    end = buf.index(b"\0", offset)
    return buf[offset:end].decode("utf-8"), end + 1


def _parse_header(buf):
    if struct.unpack_from("<I", buf, 0)[0] != EXR_MAGIC:
        raise ExrError("not an OpenEXR file (bad magic)")

    version_field = struct.unpack_from("<I", buf, 4)[0]
    if version_field & 0x200:
        raise ExrError("tiled EXR files are not supported")
    if version_field & 0x800:
        raise ExrError("deep EXR files are not supported")
    if version_field & 0x1000:
        raise ExrError("multipart EXR files are not supported")

    attrs = {}
    offset = 8
    while True:
        name, offset = _read_null_terminated(buf, offset)
        if not name:  # empty name terminates the header
            break
        _type, offset = _read_null_terminated(buf, offset)
        size = struct.unpack_from("<i", buf, offset)[0]
        offset += 4
        attrs[name] = buf[offset:offset + size]
        offset += size
    return attrs, offset


def _parse_channels(blob):
    """Channel list, in the order EXR stores them (sorted by name)."""
    channels = []
    offset = 0
    while offset < len(blob) and blob[offset] != 0:
        name, offset = _read_null_terminated(blob, offset)
        pixel_type = struct.unpack_from("<i", blob, offset)[0]
        # pLinear (1 byte) + 3 reserved + xSampling (4) + ySampling (4)
        x_sampling, y_sampling = struct.unpack_from("<ii", blob, offset + 8)
        offset += 16
        if x_sampling != 1 or y_sampling != 1:
            raise ExrError("subsampled channel %r is not supported" % name)
        if pixel_type not in _PIXEL_DTYPE:
            raise ExrError("unknown pixel type %d for channel %r" % (pixel_type, name))
        channels.append((name, pixel_type))
    return channels


def _unpredict_and_deinterleave(raw):
    """Undo the byte split and delta prediction ZIP/ZIPS apply before deflate.

    OpenEXR splits each block into even and odd bytes and delta-encodes the
    result, which is what makes float data compress at all. Both steps have to
    be undone in this order, and both are exact -- there is no tolerance here to
    tune.
    """
    data = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)

    # Delta decode. The recurrence starts at the *second* byte:
    #     t[0] = d[0]
    #     t[i] = t[i-1] + d[i] - 128
    # Subtracting 128 from d[0] as well biases every reconstructed byte by -128,
    # which flips the high bit of each one. For the exponent byte of a float that
    # is the sign bit, so the image comes back exactly negated with otherwise
    # correct magnitudes -- a failure that looks like a renderer bug, not a
    # decoder bug, which is why it is spelled out here.
    if len(data):
        data[1:] -= 128
        np.cumsum(data, out=data)
    data = (data & 0xFF).astype(np.uint8)

    # Un-split: the first half holds the even output bytes, the second the odd.
    half = (len(data) + 1) // 2
    out = np.empty(len(data), dtype=np.uint8)
    out[0::2] = data[:half]
    out[1::2] = data[half:]
    return out.tobytes()


def _decompress_rle(raw, expected):
    out = bytearray()
    i = 0
    while i < len(raw) and len(out) < expected:
        count = struct.unpack_from("<b", raw, i)[0]
        i += 1
        if count < 0:
            n = -count
            out += raw[i:i + n]
            i += n
        else:
            n = count + 1
            out += raw[i:i + 1] * n
            i += 1
    return _unpredict_and_deinterleave(bytes(out))


def load_exr(path, channel_names=("R", "G", "B")):
    """Read an EXR as float32 (h, w, len(channel_names)), top-down.

    Top-down means row 0 is the top of the image, matching what `compare.py`
    and the PNG writer expect. EXR stores INCREASING_Y (top-down) by default;
    DECREASING_Y is flipped here so callers never have to ask.
    """
    with open(path, "rb") as handle:
        buf = handle.read()

    attrs, offset = _parse_header(buf)

    for required in ("channels", "dataWindow", "compression"):
        if required not in attrs:
            raise ExrError("EXR header is missing %r" % required)

    compression = attrs["compression"][0]
    if compression not in _LINES_PER_BLOCK:
        raise ExrError(
            "compression %s is not supported by this reader; re-save as ZIP"
            % _COMPRESSION_NAMES.get(compression, compression))

    x_min, y_min, x_max, y_max = struct.unpack("<iiii", attrs["dataWindow"])
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # lineOrder: 0 = INCREASING_Y, 1 = DECREASING_Y, 2 = RANDOM_Y (tiles only).
    line_order = attrs["lineOrder"][0] if "lineOrder" in attrs else 0

    channels = _parse_channels(attrs["channels"])
    by_name = {name: (index, ptype) for index, (name, ptype) in enumerate(channels)}
    for wanted in channel_names:
        if wanted not in by_name:
            raise ExrError(
                "channel %r not in file (has %s)"
                % (wanted, ", ".join(n for n, _ in channels)))

    # One scanline holds every channel in header order, each channel contiguous.
    channel_bytes = [_PIXEL_DTYPE[ptype].itemsize * width for _, ptype in channels]
    line_bytes = sum(channel_bytes)
    channel_offsets = np.cumsum([0] + channel_bytes[:-1])

    lines_per_block = _LINES_PER_BLOCK[compression]
    num_chunks = (height + lines_per_block - 1) // lines_per_block

    offsets = struct.unpack_from("<%dQ" % num_chunks, buf, offset)

    planes = {name: np.empty((height, width), dtype=np.float32) for name in channel_names}

    for chunk_offset in offsets:
        y = struct.unpack_from("<i", buf, chunk_offset)[0]
        size = struct.unpack_from("<i", buf, chunk_offset + 4)[0]
        raw = buf[chunk_offset + 8:chunk_offset + 8 + size]

        row = y - y_min
        rows_here = min(lines_per_block, height - row)
        uncompressed_size = rows_here * line_bytes

        if size == uncompressed_size:
            # EXR stores a block verbatim whenever compressing made it bigger,
            # regardless of the compression the header names.
            block = raw
        elif compression in (COMPRESSION_ZIP, COMPRESSION_ZIPS):
            block = _unpredict_and_deinterleave(zlib.decompress(raw))
        elif compression == COMPRESSION_RLE:
            block = _decompress_rle(raw, uncompressed_size)
        else:
            block = raw

        if len(block) < uncompressed_size:
            raise ExrError(
                "short block at y=%d: got %d bytes, expected %d"
                % (y, len(block), uncompressed_size))

        for line in range(rows_here):
            base = line * line_bytes
            for name in channel_names:
                index, ptype = by_name[name]
                start = base + channel_offsets[index]
                dtype = _PIXEL_DTYPE[ptype]
                values = np.frombuffer(block, dtype=dtype, count=width,
                                       offset=start)
                planes[name][row + line] = values.astype(np.float32)

    image = np.stack([planes[name] for name in channel_names], axis=-1)
    if line_order == 1:
        image = image[::-1]
    return np.ascontiguousarray(image)
