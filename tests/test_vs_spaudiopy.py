#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:57:41 2020

@author: chris
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest

# import sys
# sys.path.append("../")   # this adds the mother folder

import spaudiopy as spa
import safpy


def test_getSHreal():
    num_dirs = 3
    rg = np.random.default_rng()
    azi = rg.uniform(0, 2*np.pi, num_dirs)
    zen = rg.uniform(0, np.pi, num_dirs)
    n_sph = 10
    a = safpy.sh.getSHreal(n_sph, np.c_[azi, zen])
    b = spa.sph.sh_matrix(n_sph, azi, zen, SH_type='real').astype(np.float32)

    assert_allclose(a, b.T, atol=10e-6)


def test_getSHcomplex():
    num_dirs = 3
    rg = np.random.default_rng()
    azi = rg.uniform(0, 2*np.pi, num_dirs)
    zen = rg.uniform(0, np.pi, num_dirs)
    n_sph = 10
    a = safpy.sh.getSHcomplex(n_sph, np.c_[azi, zen])
    b = spa.sph.sh_matrix(n_sph, azi, zen, SH_type='complex').astype(
        np.complex64)

    assert_allclose(a, b.T, atol=10e-6)


def test_vbap_gaintable_3d():
    vecs = spa.grids.load_t_design(10)
    azi, zen, r = spa.utils.cart2sph(*vecs.T)
    azi_deg = spa.utils.rad2deg(azi)
    zen_deg = spa.utils.rad2deg(zen)
    gt = safpy.vbap.generateVBAPgainTable3D(np.c_[azi_deg, zen_deg-90], 1, 1)

    assert(np.all(np.count_nonzero(gt, axis=1) <= 3))
    assert_allclose(np.sum(gt**2, axis=1), np.ones(gt.shape[0]), atol=10e-6)

test_vbap_gaintable_3d()
