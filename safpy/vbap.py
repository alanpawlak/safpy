"""Docstring VBAP."""

import numpy as np

from _safpy import ffi, lib


def generateVBAPgainTable3D(ls_dirs_deg, az_res_deg, el_res_deg,
                            omitLargeTriangles=False, enableDummies=False,
                            spread=0.):
    """
    generateVBAPgainTable3D.

    Parameters
    ----------
    ls_dirs_deg : TYPE
        DESCRIPTION.
    az_res_deg : TYPE
        DESCRIPTION.
    el_res_deg : TYPE
        DESCRIPTION.
    omitLargeTriangles : TYPE, optional
        DESCRIPTION. The default is False.
    enableDummies : TYPE, optional
        DESCRIPTION. The default is False.
    spread : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    gtable : TYPE
        DESCRIPTION.

    """
    ls_dirs_deg = np.ascontiguousarray(np.atleast_2d(ls_dirs_deg),
                                       dtype=np.float32)
    assert(np.shape(ls_dirs_deg)[1] == 2)
    num_ls = np.shape(ls_dirs_deg)[0]

    # let the compiler handle this
    # az_res_deg = int(az_res_deg)
    # el_res_deg = int(el_res_deg)
    # omitLargeTriangles = int(omitLargeTriangles)
    # enableDummies = int(enableDummies)
    # spread = float(spread)

    gtable_ptr = ffi.new("float **")
    num_gtable_out_ptr = ffi.new("int *")
    num_triangles_ptr = ffi.new("int *")

    lib.generateVBAPgainTable3D(ffi.from_buffer("float []", ls_dirs_deg),
                                num_ls,
                                az_res_deg,
                                el_res_deg,
                                omitLargeTriangles,
                                enableDummies,
                                spread,
                                gtable_ptr,
                                num_gtable_out_ptr,
                                num_triangles_ptr
                                )
    num_gtable = num_gtable_out_ptr[0]
    # num_triangles = num_triangles_ptr[0]

    gtable = np.reshape(np.array(ffi.unpack(gtable_ptr[0],
                                            num_ls*num_gtable),
                                 dtype=np.float32, ndmin=2),
                        (num_gtable, num_ls))
    return gtable


def generateVBAPgainTable3D_srcs(src_dirs_deg, ls_dirs_deg, 
                                 omitLargeTriangles=False, enableDummies=False,
                                 spread=0.):
    """
    Generates a 3-D VBAP gain table.

    Parameters
    ----------
    src_dirs_deg : np.ndarray
        Source directions in degrees; shape (S, 2).
    ls_dirs_deg : np.ndarray
        Loudspeaker directions in degrees; shape (L, 2).
    omitLargeTriangles : bool, optional
        Whether to omit large triangles, default is False.
    enableDummies : bool, optional
        Whether to enable dummies, default is False.
    spread : float, optional
        Spreading factor in degrees, default is 0.

    Returns
    -------
    gtable : np.ndarray
        The 3D VBAP gain table, shape (N_gtable, L).
    """

    # Validation and flattening of source directions array
    src_dirs_deg = np.ascontiguousarray(np.atleast_2d(src_dirs_deg), dtype=np.float32)
    assert src_dirs_deg.ndim == 2 and src_dirs_deg.shape[1] == 2, "src_dirs_deg must be a 2D array with shape (S, 2)"
    num_srcs = src_dirs_deg.shape[0]

    # Validation and flattening of loudspeaker directions array
    ls_dirs_deg = np.ascontiguousarray(np.atleast_2d(ls_dirs_deg), dtype=np.float32)
    assert ls_dirs_deg.ndim == 2 and ls_dirs_deg.shape[1] == 2, "ls_dirs_deg must be a 2D array with shape (L, 2)"
    num_ls = ls_dirs_deg.shape[0]

    # Create pointers to the flattened arrays
    src_dirs_ptr = ffi.cast("float *", src_dirs_deg.ctypes.data)
    ls_dirs_ptr = ffi.cast("float *", ls_dirs_deg.ctypes.data)

    # Preparing for C function call
    gtable_ptr = ffi.new("float **")
    num_gtable_out_ptr = ffi.new("int *")
    num_triangles_ptr = ffi.new("int *")

    # Call to C function
    lib.generateVBAPgainTable3D_srcs(src_dirs_ptr,
                                     num_srcs,
                                     ls_dirs_ptr,
                                     num_ls,
                                     omitLargeTriangles,
                                     enableDummies,
                                     spread,
                                     gtable_ptr,
                                     num_gtable_out_ptr,
                                     num_triangles_ptr
                                     )

    num_gtable = num_gtable_out_ptr[0]

    # Post-processing to get the VBAP gain table
    # Correctly unpack data from gtable_ptr
    gtable = np.reshape(np.array(ffi.unpack(gtable_ptr[0], num_ls * num_gtable),
                                 dtype=np.float32, ndmin=2),
                        (num_gtable, num_ls))

    return gtable