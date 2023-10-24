import numpy as np
import h5py
from warnings import warn
from numbers import Integral
from typing import Optional

__all__ = ["MetaArray"]


class MetaArray:
    """Array with metadata: a wrapper for Tuple(np.ndarray, dict) for `value` and
    `metadata` similar to a HDF5 Dataset. It can be converted from and to a HDF5
    dataset with method `from_hdf5` and `write_to_hdf5`.
    The returned `MetaArray` will simply wrap the input `ndarray` inside. Modifications
    to the `MetaArray` will take effect in the `ndarray` and vice versa, similar to
    pytorch's `torch.from_numpy`. (Unless argument `copy` is `True`)

    Parameters
    ----------
    value : numpy.ndarray
        The array to be wrapped.
    metadata : dict, optional
        Metadata to be wrapped, a mapping from str to str or arrays, by default None
    copy : bool, default False
        Whether to store a copy of the input array instead of the array itself


    Raises
    ------
    ValueError
        When the input `value` is not a valid numpy array.
    """

    def __init__(self, value, metadata=None, copy=False):
        if not isinstance(value, np.ndarray):
            raise ValueError("Input is not a np.ndarray.")
        if copy:
            self._value = value.copy()
        else:
            self._value = value
        self._attrs = metadata if metadata is not None else dict()

    @property
    def metadata(self):
        """Access the metadata.

        Returns
        -------
        dict
            The dictionary holding the metadata.
        """
        return self._attrs

    def __getitem__(self, key):
        """Implements the bracket indexing getter.

        Parameters
        ----------
        key : int | slice
            The usual input to index a numpy array.

        Returns
        -------
        numpy.ndarray
            Same as from indexing the underlying wrapped array.
        """
        return self._value[key]

    def __setitem__(self, key, value):
        """Mimicking the assigning behavior of a HDF5 dataset. If in
        `meta_array[...] = new_values` the `new_values` are not broadcastable to the
        original array shape, the meta_array is rebinded and no ValueError is thrown.
        """

        def is_full_slice(slice_):
            return (
                slice_.start is None
                and slice_.stop is None
                and slice_.step is None
            )

        value = np.asarray(value)
        try:
            self._value[key] = value
        except ValueError:
            if isinstance(key, slice) and is_full_slice(key):
                # unlike numpy, h5py supports reassigning self._value
                # here we use a rebind to mimic this behavior
                self._value = value
            else:
                raise

    def __repr__(self):
        return f'<MetaArray: shape {self._value.shape}, type "{self._value.dtype}">'

    @staticmethod
    def from_hdf5(
        h5dt, offset: Optional[int] = None, stride: Optional[int] = None
    ):
        """Create an instance from the content of HDF5 dataset `h5dt`. For a non-scalar
        dataset, offset and stride can be set to read in the slice
        `h5dt[offset::stride]`. For scalar dataset, `offset`
        and `stride` will be simply ignored.

        Parameters
        ----------
        h5dt : h5py.Dataset
            A HDF5 Dataset.
        offset : None | int, optional
            The offset for loading from the HDF5 file. Default is `None`.
        stride : None | int, optional
            The stride for loading from the HDF5 file. Default is `None`.

        Returns
        -------
        MetaArray
            Its value and metadata come from `h5dt`'s data and attributes, respectively.

        Raises
        ------
        ValueError
            When the input `h5dt` is not a valid h5py.Dataset.
        """
        if not isinstance(h5dt, h5py.Dataset):
            raise ValueError(
                f"Input {h5dt}'s type is {type(h5dt)}, expecting a h5py.Dataset."
            )
        if not isinstance(offset, Integral) and offset is not None:
            raise ValueError(
                f"Input `offset`'s type is {type(offset)}, expecting an `int` or `None`."
            )
        if not isinstance(stride, Integral) and stride is not None:
            raise ValueError(
                f"Input `stride`'s type is {type(stride)}, expecting an `int` or `None`."
            )
        if h5dt.ndim == 0:
            # scalar dataset, neither offset nor stride will take effect
            dt = MetaArray(h5dt[...])
        else:
            # non-scalar dataset
            dt = MetaArray(h5dt[offset::stride])
        for k, v in h5dt.attrs.items():
            dt.metadata[k] = v
        return dt

    def write_to_hdf5(self, h5_node, name=None):
        """Write the content to a HDF5 dataset at `h5_node` or `h5_node[name]` when
        `name` is not `None`.

        Parameters
        ----------
        h5_node : h5py.Group | h5py.Dataset
            When it is a `h5py.Group`, then write to `h5_node[name]`; otherwise write to
            `h5_node`.
        name : str, optional
            The name of the target Dataset when `h5_node` is a `h5py.Group`, by default
            None
        """

        # local hack for storing strings
        value_to_save = self._value
        if value_to_save.dtype == np.dtype("O"):
            value_to_save = np.asarray(str(value_to_save[()]), dtype="O")

        def overwrite(dataset_node, input_pattern="h5_node"):
            # overwrite an existing Dataset in HDF5 file
            warn(
                f"Input `{input_pattern}` correponds to existing Dataset with name {dataset_node.name}. Overwritting..."
            )
            dataset_node[...] = value_to_save
            for k, v in self.metadata.items():
                dataset_node.attrs[k] = v

        if isinstance(h5_node, h5py.Dataset):
            # h5_node correponds to existing Dataset in HDF5 file
            if name is not None:
                raise ValueError(
                    "Input `name` should be `None` when `h5_node` is a h5py.Dataset."
                )
            else:
                overwrite(h5_node)
        elif isinstance(h5_node, h5py.Group):
            # h5_node correponds to a Group in HDF5 file
            if name is None:
                raise ValueError("Input `name` cannot be `None`")
            if name in h5_node:
                overwrite(h5_node[name], "h5_node[name]")
            else:
                # create a new Dataset under h5_node
                h5_node[name] = value_to_save
                for k, v in self.metadata.items():
                    h5_node[name].attrs[k] = v
        else:
            raise ValueError(
                "Input `h5_node` should be an instance of either `h5py.Dataset` or `h5py.Group`."
            )
