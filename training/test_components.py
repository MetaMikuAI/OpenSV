import struct
import unittest

import torch

from metasv_train.components import Component, normalize
from metasv_train.format import Node


def vector(values):
    payload = struct.pack("<I", len(values)) + struct.pack(f"<{len(values)}f", *values)
    return Node(0x7FCA40FF, b"prim1\0\0\0", "prim1", payload)


class ComponentTests(unittest.TestCase):
    def test_normalizer_converts_physical_values(self):
        source = Node(0, b"", "cmpu0", b"", children=[vector([2., 4.]), vector([8., 12.]), vector([-1., 1.])])
        raw = torch.tensor([[3., 8.]])
        torch.testing.assert_close(normalize(source, normalize(source, raw), True), raw)

    def test_embedding_mapping_uses_labels_not_column_number(self):
        component = Component.__new__(Component)
        torch.nn.Module.__init__(component)
        component.teacher_embeddings = {"phone": torch.tensor([[1., 2.], [3., 4.]])}
        component.embeddings = torch.nn.ParameterDict({"phone": torch.nn.Parameter(torch.tensor([[20., 10.], [40., 30.]]))})
        component.maps = {"phone": [1, 0]}
        observed = torch.tensor([[1., 3.], [2., 4.]])
        actual = component.lookup("phone", observed)
        torch.testing.assert_close(actual, torch.tensor([[10., 30.], [20., 40.]]))
        actual.sum().backward()
        self.assertIsNotNone(component.embeddings["phone"].grad)
        component.maps["phone"] = [-1, 0]
        with self.assertRaisesRegex(ValueError, "absent"):
            component.lookup("phone", observed)


if __name__ == "__main__":
    unittest.main()
