import pytest
import numpy as np
from .meta_array import *
from tempfile import TemporaryFile
import h5py


def test_meta_array():
    value = np.random.rand(7, 5)
    ma = MetaArray(value, metadata={"test": "metadata_no1"})
    ma[:, 2:4] = 5.0
    value[:, 2:4] = 5.0
    with TemporaryFile() as fp:
        with h5py.File(fp, "w") as hdf_root:
            ma.write_to_hdf5(hdf_root, "test_dt")
        with h5py.File(fp, "r") as hdf_root:
            ma2 = MetaArray.from_hdf5(hdf_root["test_dt"])
            ma2_strided = MetaArray.from_hdf5(hdf_root["test_dt"], stride=2)
            ma2_offsetted = MetaArray.from_hdf5(hdf_root["test_dt"], offset=1)
            ma2_strided_offsetted = MetaArray.from_hdf5(
                hdf_root["test_dt"], stride=3, offset=2
            )
    assert np.allclose(ma2[1:, 1:3], value[1:, 1:3])
    assert ma2.metadata["test"] == "metadata_no1"
    assert np.allclose(ma2_strided[:], value[::2])
    assert np.allclose(ma2_offsetted[:], value[1:])
    assert np.allclose(ma2_strided_offsetted[:], value[2::3])


def test_str_meta_array():
    # test both the support for a scalar and for a string
    value = np.asarray("asadfghjk", dtype="O")
    ma = MetaArray(value)
    with TemporaryFile() as fp:
        with h5py.File(fp, "w") as hdf_root:
            ma.write_to_hdf5(hdf_root, "test_dt")
        with h5py.File(fp, "r") as hdf_root:
            ma2 = MetaArray.from_hdf5(hdf_root["test_dt"])
            ma2_offsetted = MetaArray.from_hdf5(hdf_root["test_dt"], offset=2)
            ma2_strided = MetaArray.from_hdf5(hdf_root["test_dt"], stride=2)
            ma2_strided_offsetted = MetaArray.from_hdf5(
                hdf_root["test_dt"], stride=3, offset=2
            )
    assert (
        ma2[()]
        == ma2_offsetted[()]
        == ma2_strided[()]
        == ma2_strided_offsetted[()]
        == "asadfghjk".encode("utf8")
    )
