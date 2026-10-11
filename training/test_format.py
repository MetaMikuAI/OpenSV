import struct
import unittest

import numpy as np

from metasv_train.format import Model, SEEDS, tensor, type_hash, replace_tensor


class DenseExportTests(unittest.TestCase):
    def test_sparse_matrix_can_gain_new_weights(self):
        payload = struct.pack("<5I2H3I2f", 2, 2, 1, 1, 2, 0, 1, 0, 1, 2, 1., 2.)
        data = struct.pack("<III8sII", 0x7FCA00FF, 1, 0x7FCA40FF, b"prim3\0\0\0", 0, len(payload)) + payload
        model = Model(data)
        model.make_dense([model.root])
        values = tensor(model.root)
        values[0, 1] = 0.125
        replace_tensor(model.root, values)
        np.testing.assert_array_equal(tensor(Model(model.to_bytes()).root), values)

    def test_quantized_matrix_becomes_editable_without_rounding(self):
        for version in (1, 2):
            with self.subTest(version=version):
                tag = b"prim4\0\0\0" if version == 1 else type_hash("prim4", SEEDS[3]).to_bytes(8, "little")
                count = 0
                if version == 2:
                    low, high = struct.unpack("<II", tag)
                    count = low ^ high ^ 0xAC5E7BD5
                payload = struct.pack("<IIII2fII4b", 8, 0, 0, 2, 1., 2., 2, 2, 32, 64, -32, -64)
                data = struct.pack("<III8sII", 0x7FCA00FF, version, 0x7FCA40FF, tag, count, len(payload)) + payload
                model = Model(data)
                before = tensor(model.root)
                model.make_dense([model.root])
                np.testing.assert_array_equal(tensor(model.root), before)
                values = before + np.float32(0.000013)
                replace_tensor(model.root, values)
                reloaded = Model(model.to_bytes())
                self.assertEqual(reloaded.root.type, "prim2")
                np.testing.assert_array_equal(tensor(reloaded.root), values)


if __name__ == "__main__":
    unittest.main()
