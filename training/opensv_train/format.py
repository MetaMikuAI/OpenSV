"""DNNI 读取与保持布局的参数写入。"""

from dataclasses import dataclass, field
from pathlib import Path
import struct

import numpy as np


SEEDS = (0x0DCD59189D5A0F24, 0x59346D79970CA21E, 0xCF3519B773B767BF,
         0x562C8E41FB7FBEE2, 0xBD6D25457E1ED24E, 0x0123456789ABCDEF,
         0x76543210FEDCBA98, 0x02468ACEECA86420, 0xECA864202468ACEE,
         0x70556F5965766947, 0x5A4D5A4D5A4D5A4D, 0x000000626D6F6379)
TYPES = ("prim0 prim1 prim2 prim3 prim4 prim5 modm0 modl0 modl1 modl3 modl4 "
         "modl6 moda0 moda1 moda2 moda3 moda4 moda5 moda7 _gnc1v0 _ncwnv0 "
         "cmpg1 cmpu0 cmpu1 _vocfv1 _vocfv2 _ppusv0 _ppdsv0 _psv2 _rldtg0 "
         "_rldms0 _stbkv1 _vqctx1 _didsv0 _ftmfv2 _ftmfv3 _dctov0").split()
MATRICES = {"prim0", "prim2", "prim3", "prim4", "prim5"}


def type_hash(name, seed):
    for byte in name.encode("ascii"):
        seed = ((seed ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return seed


HASH_TYPES = {type_hash(name, seed): name for name in TYPES for seed in SEEDS}


@dataclass
class Node:
    marker: int
    tag: bytes
    type: str
    payload: bytes
    offset: int = 0
    children: list = field(default_factory=list)
    index: int = 0


class Model:
    def __init__(self, data):
        if len(data) < 28 or len(data) > 512 * 1024 * 1024:
            raise ValueError("DNNI size outside supported bounds")
        magic, self.version = struct.unpack_from("<II", data)
        if magic != 0x7FCA00FF or self.version not in (1, 2):
            raise ValueError("Unsupported DNNI signature/version")
        self.nodes = []
        self.root, end = self._parse(data, 8, 0)
        if end != len(data):
            raise ValueError("Trailing bytes after root")

    @classmethod
    def read(cls, path):
        return cls(Path(path).read_bytes())

    def _parse(self, data, pos, depth):
        if depth > 256 or len(self.nodes) >= 1000000 or pos + 20 > len(data):
            raise ValueError(f"Invalid node bounds at {pos:#x}")
        marker, tag, count, size = struct.unpack_from("<I8sII", data, pos)
        if marker not in (0x7FCA40FF, 0x7FCA41FF):
            raise ValueError(f"Invalid marker at {pos:#x}")
        if self.version == 2:
            low, high = struct.unpack("<II", tag)
            count ^= low ^ high ^ 0xAC5E7BD5
            if count % 3:
                raise ValueError("Invalid v2 child count")
            count //= 3
            name = HASH_TYPES.get(int.from_bytes(tag, "little"), "unknown:" + tag.hex())
        else:
            raw = tag.rstrip(b"\0")
            if not raw or any(x < 32 or x > 126 for x in raw):
                raise ValueError("Invalid v1 type")
            name = raw.decode("ascii")
        end = pos + 20 + size
        if end > len(data) or count > (len(data) - end) // 20:
            raise ValueError("Payload or children exceed file")
        node = Node(marker, tag, name, data[pos + 20:end], pos, index=len(self.nodes))
        self.nodes.append(node)
        for _ in range(count):
            child, end = self._parse(data, end, depth + 1)
            node.children.append(child)
        return node, end

    def to_bytes(self):
        def encode(node):
            count = len(node.children)
            if self.version == 2:
                low, high = struct.unpack("<II", node.tag)
                count = (count * 3) ^ low ^ high ^ 0xAC5E7BD5
            header = struct.pack("<I8sII", node.marker, node.tag, count, len(node.payload))
            return header + node.payload + b"".join(encode(n) for n in node.children)
        return struct.pack("<II", 0x7FCA00FF, self.version) + encode(self.root)

    def write_new(self, path):
        with Path(path).open("xb") as output:
            output.write(self.to_bytes())

    def make_dense(self, nodes):
        """将指定矩阵转为浮点稠密存储，保留版本与类型哈希种子。"""
        for node in nodes:
            if node.type not in MATRICES:
                raise ValueError("Expected a matrix node")
            if node.type in ("prim0", "prim2"):
                continue
            values = tensor(node)
            if self.version == 2:
                seed = next((s for s in SEEDS if type_hash(node.type, s) == int.from_bytes(node.tag, "little")), None)
                if seed is None:
                    raise ValueError("Unknown matrix type hash seed")
                node.tag = type_hash("prim2", seed).to_bytes(8, "little")
            else:
                node.tag = b"prim2\0\0\0"
            node.type = "prim2"
            node.payload = struct.pack("<II", *values.shape) + values.astype("<f4").tobytes()


def _matrix_layout(node):
    payload = node.payload
    quantized = node.type in ("prim4", "prim5")
    sparse = node.type in ("prim3", "prim5")
    bits, offset, scales = 32, 0, None
    if quantized:
        bits, input_quant, residual, count = struct.unpack_from("<IIII", payload)
        if bits not in (8, 16) or residual:
            raise ValueError("Unsupported weight quantization")
        offset = 16 + count * 4
        scales = np.frombuffer(payload, dtype="<f4", count=count, offset=16)
    rows, columns = struct.unpack_from("<II", payload, offset)
    if not rows or not columns or rows * columns > 128 * 1024 * 1024:
        raise ValueError("Invalid matrix dimensions")
    if quantized and (len(scales) != rows or not np.isfinite(scales).all()):
        raise ValueError("Invalid row scales")
    dtype = np.dtype("<f4" if not quantized else ("i1" if bits == 8 else "<i2"))
    if not sparse:
        start = offset + 8
        if len(payload) != start + rows * columns * dtype.itemsize:
            raise ValueError("Matrix payload size mismatch")
        return rows, columns, bits, scales, dtype, start, None
    br, bc, count = struct.unpack_from("<III", payload, offset + 8)
    if not br or not bc or rows % br or columns % bc:
        raise ValueError("Unsupported sparse block dimensions")
    index_start = offset + 20
    indices = np.frombuffer(payload, dtype="<u2", count=count, offset=index_start)
    pointers = np.frombuffer(payload, dtype="<u4", count=rows // br + 1,
                             offset=index_start + count * 2)
    start = index_start + count * 2 + len(pointers) * 4
    if (len(payload) != start + count * br * bc * dtype.itemsize or pointers[0] != 0
            or pointers[-1] != count or np.any(pointers[1:] < pointers[:-1])
            or np.any(indices >= columns // bc)):
        raise ValueError("Invalid sparse arrays")
    return rows, columns, bits, scales, dtype, start, (br, bc, indices, pointers)


def tensor(node):
    if node.type == "prim1":
        count, = struct.unpack_from("<I", node.payload)
        if len(node.payload) != 4 + count * 4:
            raise ValueError("Vector payload size mismatch")
        result = np.frombuffer(node.payload, dtype="<f4", offset=4).copy()
    elif node.type in MATRICES:
        rows, cols, bits, scales, dtype, start, sparse = _matrix_layout(node)
        raw = np.frombuffer(node.payload, dtype=dtype, offset=start).astype(np.float32)
        if sparse is None:
            result = raw.reshape(cols, rows).T.copy() if scales is not None else raw.reshape(rows, cols)
        else:
            br, bc, indices, pointers = sparse
            blocks = raw.reshape(-1, bc, br).transpose(0, 2, 1)
            result = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows // br):
                for i in range(int(pointers[r]), int(pointers[r + 1])):
                    c = int(indices[i])
                    result[r * br:(r + 1) * br, c * bc:(c + 1) * bc] += blocks[i]
        if scales is not None:
            result = (result / np.float32(2 ** (bits - 1))) * scales[:, None]
    else:
        raise ValueError(f"Not a tensor: {node.type}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite tensor")
    return result


def replace_tensor(node, values):
    """保持原布局；拒绝量化溢出和新增稀疏块。"""
    values = np.asarray(values, dtype=np.float32)
    original = tensor(node)
    if values.shape != original.shape or not np.isfinite(values).all():
        raise ValueError("Replacement shape or values invalid")
    if np.array_equal(values, original):
        return
    if node.type == "prim1":
        node.payload = node.payload[:4] + values.astype("<f4").tobytes()
        return
    rows, cols, bits, scales, dtype, start, sparse = _matrix_layout(node)
    encoded = values.copy()
    if scales is not None:
        if np.any(scales == 0):
            raise ValueError("Cannot rewrite zero-scale quantized rows")
        encoded = np.rint(values / scales[:, None] * np.float32(2 ** (bits - 1)))
        limit = np.iinfo(dtype)
        if np.any(encoded < limit.min) or np.any(encoded > limit.max):
            raise ValueError("Replacement exceeds existing quantization range")
    if sparse is None:
        raw = encoded.T if scales is not None else encoded
    else:
        br, bc, indices, pointers = sparse
        mask = np.zeros((rows, cols), dtype=bool)
        blocks = []
        for r in range(rows // br):
            seen = set()
            for i in range(int(pointers[r]), int(pointers[r + 1])):
                c = int(indices[i])
                if c in seen:
                    raise ValueError("Rewriting duplicate sparse blocks is unsupported")
                seen.add(c)
                region = np.s_[r * br:(r + 1) * br, c * bc:(c + 1) * bc]
                mask[region] = True
                blocks.append(encoded[region].T.flatten())
        if np.any(values[~mask] != 0):
            raise ValueError("Replacement creates weights outside original sparse blocks")
        raw = np.concatenate(blocks) if blocks else np.empty(0)
    node.payload = node.payload[:start] + raw.astype(dtype).tobytes()


def inventory(model):
    result = []
    for node in model.nodes:
        row = {"index": node.index, "offset": node.offset, "type": node.type,
               "payload_bytes": len(node.payload), "children": [c.index for c in node.children]}
        if node.type in MATRICES or node.type == "prim1":
            try:
                row["shape"] = list(tensor(node).shape)
            except (ValueError, struct.error) as error:
                row["tensor_error"] = str(error)
        result.append(row)
    return result
