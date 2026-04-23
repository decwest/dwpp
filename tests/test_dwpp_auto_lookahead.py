import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pure_pursuit as pp  # noqa: E402
import benchmark_compare_three_methods as benchmark  # noqa: E402


class DwppAutoLookAheadTest(unittest.TestCase):
    def test_forward_point_indices_skip_points_behind_robot(self) -> None:
        current_pose = np.array([0.0, 0.0, 0.0])
        path = np.array([
            [0.0, 0.0],
            [-0.3, 0.0],
            [0.2, 0.0],
            [0.4, 0.1],
        ])

        forward_indices = pp.calc_forward_point_indices(current_pose, current_idx=0, path=path)

        np.testing.assert_array_equal(forward_indices, np.array([2, 3]))

    def test_dwpp_selects_first_candidate_with_intersection(self) -> None:
        current_pose = np.array([0.0, 0.0, 0.0])
        current_velocity = np.array([0.0, 0.0])
        path = np.array([
            [0.0, 0.0],
            [0.15, 0.15],
            [0.50, 0.05],
        ])
        path_distances = pp.calc_path_distances(path)

        next_velocity_ref, look_ahead_pos, curvature, regulated_v = pp.calc_dwpp_velocity_with_auto_look_ahead(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=0,
            path=path,
            path_distances=path_distances,
            is_accel=True,
            use_regulated_velocity=True,
        )

        forward_indices = pp.calc_forward_point_indices(current_pose, current_idx=0, path=path)
        first_candidate_idx = int(forward_indices[0])
        first_candidate_pos = path[first_candidate_idx, :2]
        first_candidate_curvature = pp.calc_curvature_to_point(current_pose, first_candidate_pos)
        first_candidate_regulated_v = pp.calc_regulated_translational_velocity(first_candidate_curvature)
        first_candidate_velocity, has_intersection, _ = pp.calc_velocity_command_for_curvature(
            current_velocity=current_velocity,
            regulated_v=first_candidate_regulated_v,
            curvature=first_candidate_curvature,
            prefer_high_speed=True,
        )
        self.assertTrue(has_intersection)

        later_candidate_idx = int(forward_indices[1])
        later_candidate_pos = path[later_candidate_idx, :2]
        later_candidate_curvature = pp.calc_curvature_to_point(current_pose, later_candidate_pos)
        later_candidate_regulated_v = pp.calc_regulated_translational_velocity(later_candidate_curvature)
        later_candidate_velocity, later_has_intersection, _ = pp.calc_velocity_command_for_curvature(
            current_velocity=current_velocity,
            regulated_v=later_candidate_regulated_v,
            curvature=later_candidate_curvature,
            prefer_high_speed=True,
        )
        self.assertTrue(later_has_intersection)
        self.assertGreater(float(later_candidate_velocity[0]), float(first_candidate_velocity[0]))

        np.testing.assert_allclose(look_ahead_pos, first_candidate_pos)
        np.testing.assert_allclose(next_velocity_ref, first_candidate_velocity)
        self.assertAlmostEqual(float(curvature), float(first_candidate_curvature))
        self.assertAlmostEqual(float(regulated_v), float(first_candidate_regulated_v))

    def test_dwpp_skips_candidates_closer_than_vdt(self) -> None:
        current_pose = np.array([0.0, 0.0, 0.0])
        current_velocity = np.array([0.2, 0.0])
        path = np.array([
            [0.0, 0.0],
            [0.005, 0.0],
            [0.020, 0.0],
        ])
        path_distances = pp.calc_path_distances(path)

        min_look_ahead_distance = pp.calc_min_look_ahead_distance_from_velocity(current_velocity)
        self.assertAlmostEqual(min_look_ahead_distance, 0.01)

        forward_indices = pp.calc_forward_point_indices(current_pose, current_idx=0, path=path)
        filtered_indices = pp.apply_min_look_ahead_distance_to_indices(
            current_idx=0,
            path_distances=path_distances,
            candidate_indices=forward_indices,
            min_look_ahead_distance=min_look_ahead_distance,
        )
        np.testing.assert_array_equal(filtered_indices, np.array([2]))

        next_velocity_ref, look_ahead_pos, curvature, regulated_v = pp.calc_dwpp_velocity_with_auto_look_ahead(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=0,
            path=path,
            path_distances=path_distances,
            is_accel=True,
            use_regulated_velocity=True,
        )

        np.testing.assert_allclose(look_ahead_pos, path[2, :2])
        self.assertAlmostEqual(float(curvature), 0.0)
        self.assertAlmostEqual(float(regulated_v), float(pp.V_MAX))
        self.assertGreater(float(next_velocity_ref[0]), float(current_velocity[0]))
        self.assertAlmostEqual(float(next_velocity_ref[1]), 0.0)

    def test_dwpp_uses_last_candidate_when_no_intersection_exists(self) -> None:
        current_pose = np.array([0.0, 0.0, 0.0])
        current_velocity = np.array([0.1, 0.6])
        path = np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.1, 0.1],
        ])
        path_distances = pp.calc_path_distances(path)

        next_velocity_ref, look_ahead_pos, curvature, regulated_v = pp.calc_dwpp_velocity_with_auto_look_ahead(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=0,
            path=path,
            path_distances=path_distances,
            is_accel=True,
            use_regulated_velocity=False,
        )

        first_curvature = pp.calc_curvature_to_point(current_pose, path[1, :2])
        first_velocity, first_has_intersection, _ = pp.calc_velocity_command_for_curvature(
            current_velocity=current_velocity,
            regulated_v=pp.V_MAX,
            curvature=first_curvature,
            prefer_high_speed=True,
        )
        self.assertFalse(first_has_intersection)

        last_curvature = pp.calc_curvature_to_point(current_pose, path[2, :2])
        last_velocity, last_has_intersection, _ = pp.calc_velocity_command_for_curvature(
            current_velocity=current_velocity,
            regulated_v=pp.V_MAX,
            curvature=last_curvature,
            prefer_high_speed=True,
        )
        self.assertFalse(last_has_intersection)

        np.testing.assert_allclose(look_ahead_pos, path[2, :2])
        np.testing.assert_allclose(next_velocity_ref, last_velocity)
        self.assertFalse(np.allclose(first_velocity, last_velocity))
        self.assertAlmostEqual(float(curvature), float(last_curvature))
        self.assertAlmostEqual(float(regulated_v), float(pp.V_MAX))

    def test_calc_desired_velocity_vector_omni_is_finite_at_zero_distance(self) -> None:
        current_pose = np.array([1.0, 2.0, 0.0])
        look_ahead_pose = np.array([1.0, 2.0, np.pi / 2.0])

        desired_velocity = pp.calc_desired_velocity_vector_omnidirectional(current_pose, look_ahead_pose)

        self.assertTrue(np.all(np.isfinite(desired_velocity)))
        np.testing.assert_allclose(desired_velocity[:2], np.array([0.0, 0.0]))
        self.assertAlmostEqual(float(desired_velocity[2]), float(pp.W_MAX))

    def test_pure_pursuit_dwpp_fixed_matches_legacy_single_lookahead_logic(self) -> None:
        current_pose = np.array([0.0, 0.0, 0.0])
        current_velocity = np.array([0.0, 0.0])
        path = np.array([
            [0.0, 0.0],
            [0.2, 0.1],
            [0.5, 0.2],
        ])

        current_idx = pp.calc_index(current_pose, path)
        path_distances = pp.calc_path_distances(path)
        look_ahead_distance = pp.calc_look_ahead_distance(current_velocity, "dwpp_fixed")
        expected_curvature, expected_look_ahead_pos = pp.calc_curvature_to_look_ahead_position(
            current_pose,
            current_idx,
            path,
            path_distances,
            look_ahead_distance,
        )
        expected_regulated_v = pp.V_MAX
        expected_is_accel = pp.decide_accel_or_decel(current_idx, path_distances)
        expected_velocity = pp.calc_optimal_velocity_considering_dynamic_window(
            current_velocity,
            expected_regulated_v,
            expected_curvature,
            expected_is_accel,
        )

        next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v = pp.pure_pursuit(
            current_pose=current_pose,
            current_velocity=current_velocity,
            path=path,
            method_name="dwpp_fixed",
        )

        np.testing.assert_allclose(look_ahead_pos, expected_look_ahead_pos)
        np.testing.assert_allclose(next_velocity_ref, expected_velocity)
        self.assertEqual(break_constraints_flag, [False, False])
        self.assertAlmostEqual(float(curvature), float(expected_curvature))
        self.assertAlmostEqual(float(regulated_v), float(expected_regulated_v))

    def test_pure_pursuit_dwpp_uses_vmax_when_vreg_is_off(self) -> None:
        current_pose = np.array([0.0, 0.0, 0.0])
        current_velocity = np.array([0.0, 0.0])
        path = np.array([
            [0.0, 0.0],
            [0.15, 0.15],
            [0.50, 0.05],
        ])

        next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v = pp.pure_pursuit(
            current_pose=current_pose,
            current_velocity=current_velocity,
            path=path,
            method_name="dwpp",
        )

        path_distances = pp.calc_path_distances(path)
        expected_velocity, expected_look_ahead_pos, expected_curvature, expected_regulated_v = pp.calc_dwpp_velocity_with_auto_look_ahead(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=0,
            path=path,
            path_distances=path_distances,
            is_accel=True,
            use_regulated_velocity=False,
        )

        np.testing.assert_allclose(next_velocity_ref, expected_velocity)
        np.testing.assert_allclose(look_ahead_pos, expected_look_ahead_pos)
        self.assertEqual(break_constraints_flag, [False, False])
        self.assertAlmostEqual(float(curvature), float(expected_curvature))
        self.assertAlmostEqual(float(regulated_v), float(expected_regulated_v))

    def test_calc_vw_line_segment_in_bounds(self) -> None:
        segment = benchmark.calc_vw_line_segment_in_bounds(
            curvature=2.0,
            v_min=0.0,
            v_max=0.22,
            w_min=-0.6,
            w_max=0.6,
        )

        np.testing.assert_allclose(segment, np.array([
            [0.0, 0.0],
            [0.22, 0.44],
        ]))

    def test_calc_dynamic_window_curvature_range(self) -> None:
        finite_kappa_min, finite_kappa_max = benchmark.calc_dynamic_window_curvature_range(
            dw_vmin=0.10,
            dw_vmax=0.20,
            dw_wmin=-0.30,
            dw_wmax=0.50,
        )
        self.assertAlmostEqual(finite_kappa_min, -3.0)
        self.assertAlmostEqual(finite_kappa_max, 5.0)

        inf_kappa_min, inf_kappa_max = benchmark.calc_dynamic_window_curvature_range(
            dw_vmin=0.0,
            dw_vmax=0.20,
            dw_wmin=-0.30,
            dw_wmax=0.50,
        )
        self.assertTrue(np.isneginf(inf_kappa_min))
        self.assertTrue(np.isposinf(inf_kappa_max))


if __name__ == "__main__":
    unittest.main()
