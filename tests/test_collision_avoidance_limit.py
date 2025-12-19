"""Tests for collision_avoidance_limit.py."""

import itertools

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.limits.collision_avoidance_limit import (
    Contact,
    compute_contact_normal_jacobian,
)
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_geom_pairs_strings(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="ball" name="ball"/>
              <geom name="g1" type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge" range="0 1.57"/>
                <geom name="g2" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body>
              <joint type="hinge" name="hinge2" range="0 1.57"/>
              <geom name="g3" type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = CollisionAvoidanceLimit(model=model, geom_pairs=[(["g1"], ["g3"])])
        self.assertListEqual(limit.geom_id_pairs, [(0, 2)])

    def test_dimensions(self):
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Check that non-colliding geoms are correctly filtered out and that we have
        # the right number of max expected contacts.
        g1_coll = [
            g
            for g in g1
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        g2_coll = [
            g
            for g in g2
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)
        assert G is not None and h is not None

        # The upper bound should always be >= relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check that the inequality constraint dimensions are valid.
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        model = load_robot_description("ur5e_mj_description")
        nv = model.nv

        # Options necessary to obtain separation normal + dense matrices.
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

        # Remove unnecessary constraints.
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT

        # Set contact dimensionality to 1 (normals only).
        model.geom_condim[:] = 1

        data = mujoco.MjData(model)

        # Handcrafted qpos with multiple contacts.
        qpos_coll = np.asarray([-1.5708, -1.5708, 3.01632, -1.5708, -1.5708, 0])
        data.qpos = qpos_coll
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 1)

        for i in range(data.ncon):
            # Get MuJoCo's contact normal jacobian.
            contact = data.contact[i]
            start_idx = contact.efc_address * nv
            end_idx = start_idx + nv
            efc_J = data.efc_J[start_idx:end_idx]

            # Compute the contact Jacobian manually.
            normal = contact.frame[:3]
            dist = contact.dist
            fromto = np.empty((6,), dtype=np.float64)
            fromto[3:] = contact.pos - 0.5 * dist * normal
            fromto[:3] = contact.pos + 0.5 * dist * normal
            contact = Contact(
                dist=contact.dist,
                fromto=fromto,
                geom1=contact.geom1,
                geom2=contact.geom2,
                distmax=np.inf,
            )
            jac = compute_contact_normal_jacobian(model, data, contact)

            np.testing.assert_allclose(jac, efc_J, atol=1e-7)

    def test_qp_upper_bound_branches_with_active_contact(self):
        """Exercise both upper-bound branches with an active contact.

        We place two touching spheres (hi_bound_dist=0) and use a large detection
        distance so the contact is active. With a positive minimum_distance, h equals
        the bound_relaxation (else branch); with a negative minimum_distance, h exceeds
        the relaxation (if branch). G remains identical across both runs.
        """
        xml = """
        <mujoco>
        <worldbody>
            <body name="left">
            <joint type="free"/>
            <geom name="gL" type="sphere" size="0.01" pos="0 0 0"/>
            </body>
            <body name="right">
            <joint type="free"/>
            <geom name="gR" type="sphere" size="0.01" pos="0.02 0 0"/>
            </body>
        </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        cfg = Configuration(model)
        cfg.update(np.zeros(model.nq))

        detect = 0.5
        dt = 1e-3

        # hi_bound_dist (0) <= minimum_distance_from_collisions (positive).
        limit_else = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(["gL"], ["gR"])],
            collision_detection_distance=detect,
            minimum_distance_from_collisions=5e-4,  # 0 <= 5e-4 -> else
            bound_relaxation=1e-4,
        )
        G_else, h_else = limit_else.compute_qp_inequalities(cfg, dt)
        assert G_else is not None and h_else is not None

        # All rows that correspond to the active pair should be exactly the relaxation.
        self.assertGreater(h_else.size, 0)
        self.assertTrue(np.any(np.isfinite(h_else)))
        self.assertTrue(np.all(h_else[np.isfinite(h_else)] >= 1e-4 - 1e-12))

        # hi_bound_dist (0) > minimum_distance_from_collisions (negative).
        limit_if = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(["gL"], ["gR"])],
            collision_detection_distance=detect,
            minimum_distance_from_collisions=-5e-3,  # 0 > -5e-3 -> if
            bound_relaxation=1e-4,
            gain=0.85,
        )
        G_if, h_if = limit_if.compute_qp_inequalities(cfg, dt)
        assert G_if is not None and h_if is not None

        # h should be strictly larger than relaxation now (gain*dist/dt + relaxation)
        self.assertGreater(h_if.size, 0)
        self.assertTrue(np.any(h_if[np.isfinite(h_if)] > 1e-4 + 1e-12))

        # G should be identical for the same geometry, only h changes.
        self.assertEqual(G_else.shape, G_if.shape)
        np.testing.assert_allclose(G_else, G_if, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
