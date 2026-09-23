import unittest

import numpy as np

from data.scripts.build_yolo_dataset import body_pixels, yolo_box


class YoloDatasetLabelTests(unittest.TestCase):
    def test_segmentation_body_id_strips_link_bits(self):
        segmentation = np.array([[3 | (7 << 24), -1, 4]], dtype=np.int32)
        pixels = body_pixels(segmentation, 3)
        self.assertEqual(pixels.tolist(), [[0, 0]])

    def test_yolo_box_is_normalized_and_positive(self):
        pixels = np.array([[10, 20], [30, 50]], dtype=np.int32)
        box = yolo_box(pixels, 100, 100)
        self.assertEqual(box, (0.35, 0.2, 0.3, 0.2))

    def test_empty_visible_body_is_rejected(self):
        self.assertIsNone(yolo_box(np.empty((0, 2), dtype=np.int32), 640, 480))


if __name__ == "__main__":
    unittest.main()
