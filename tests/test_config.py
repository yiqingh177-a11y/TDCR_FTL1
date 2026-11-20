"""The single JSON interface validates before numerical computation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from continuum_ftl.config import load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/example.json"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path)

    def test_example_and_read_only_arrays(self):
        config = self.load(self.data)
        self.assertEqual(config.robot.section_count, 3)
        self.assertFalse(config.robot.base_position_m.flags.writeable)

    def test_invalid_dimensions_limits_shapes_and_settings(self):
        variants = []

        def add_change(group, field, value):
            data = copy.deepcopy(self.data)
            data[group][field] = value
            variants.append(data)

        add_change("robot", "sections", [])
        add_change("robot", "base_position_m", [0, 0])
        add_change("robot", "initial_configuration_rad", [[0, 0]])
        add_change("robot", "initial_configuration_rad", [[2, 0], [0, 0], [0, 0]])
        add_change("target", "tolerance_m", 0)
        add_change("target", "position_m", [float("nan"), 0, 0])
        add_change("solver", "max_iterations", True)
        add_change("sampling", "max_arc_spacing_m", -0.01)
        add_change("environment", "safety_margin_m", -1)
        add_change("environment", "workspace_bounds_m", [[1, -1]] * 3)
        add_change(
            "environment", "obstacles", [{"kind": "plane", "name": "bad", "center_m": [0, 0, 0]}]
        )
        add_change("visualization", "obstacle_alpha", 1.1)
        add_change("visualization", "section_colors", [])
        add_change("robot", "unknown", 1)
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)
        for key in ("length_m", "radius_m", "max_bend_rad"):
            data = copy.deepcopy(self.data)
            data["robot"]["sections"][0][key] = -1
            with self.assertRaises(ValueError):
                self.load(data)


if __name__ == "__main__":
    unittest.main()
