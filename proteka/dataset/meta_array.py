import numpy as np
import h5py
from warnings import warn

__all__ = ["MetaArray"]


class MetaArray:
    """Array with metadata: a wrapper for Tuple(np.ndarray, dict) similar to a HDF5 Dataset. It can be converted from and to a HDF5 dataset.
    Warning: the MetaArray is a wrapper, the underlying `value` array is not copied.
    """

    def __init__(self, value, attrs={}):
        if not isinstance(value, np.ndarray):
            raise ValueError("Input is not a np.ndarray.")
        self._value = value
        self._attrs = attrs

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, key):
        return self._value[key]

    def __setitem__(self, key, value):
        """Mimicking the assigning behavior of a HDF5 dataset. If in `meta_array[...] = new_values` the `new_values` are not broadcastable to the original array shape, the meta_array is rebinded and no ValueError is thrown."""

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
    def from_hdf5(h5dt):
        """Create an instance from the content of HDF5 dataset `h5dt`."""
        if not isinstance(h5dt, h5py.Dataset):
            raise ValueError(
                f"Input {h5dt}'s type is {type(h5dt)}, expecting a h5py.Dataset."
            )
        dt = MetaArray(h5dt[...])
        for k, v in h5dt.attrs.items():
            dt.attrs[k] = v
        return dt

    def write_to_hdf5(self, h5_node, name=None):
        """Write the content to a HDF5 dataset at `h5_node` or `h5_node[name]` when `name` is not `None`."""

        def overwrite(dataset_node, input_pattern="h5_node"):
            # overwrite an existing Dataset in HDF5 file
            warn(
                f"Input `{input_pattern}` correponds to existing Dataset with name {dataset_node.name}. Overwritting..."
            )
            dataset_node[...] = self._value
            for k, v in self.attrs.items():
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
                h5_node[name] = self._value
                for k, v in self.attrs.items():
                    h5_node[name].attrs[k] = v
        else:
            raise ValueError(
                "Input `h5_node` should be an instance of either `h5py.Dataset` or `h5py.Group`."
            )
