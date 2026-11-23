"""Sampled sweeps must respect accepted frames and replay time in 2D and 3D."""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continuum_ftl.sweep_visualization import CentrelineSweep


class SweepVisualizationTests(unittest.TestCase):
    def test_sweep_renders_in_both_dimensions_and_replay_clears_future(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d" if dimension == 3 else None)
                a = np.zeros((2, dimension))
                a[-1, -1] = 1
                b = a.copy()
                b[-1, 0] = 0.2
                sweep = CentrelineSweep(ax, (a, b), (True, True), dimension=dimension)
                sweep.draw_frame(1)
                self.assertGreater(len(sweep.frames[1]), 0)
                fig.canvas.draw()
                sweep.draw_frame(0)
                fig.canvas.draw()
                if dimension == 2:
                    self.assertEqual(len(sweep.artist.get_paths()), 0)
                plt.close(fig)

    def test_rejected_frame_and_gap_are_not_added_to_sweep(self):
        fig, ax = plt.subplots()
        curve = np.array([[0.0, 0.0], [0.0, 1.0]])
        sweep = CentrelineSweep(
            ax,
            (curve, curve + 1, curve + 2),
            (True, False, True),
            dimension=2,
        )
        self.assertTrue(all(len(frame) == 0 for frame in sweep.frames))
        sweep.draw_frame(2)
        fig.canvas.draw()
        plt.close(fig)
