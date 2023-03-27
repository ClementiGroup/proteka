import numpy as np
import h5py
from warnings import warn

__all__ = ["MetaArray"]


class MetaArray:
    """Array with metadata: a wrapper for Tuple(np.ndarray, dict) similar to a HDF5
    Dataset. It can be converted from and to a HDF5 dataset.
    """

    def __init__(self, value, metadata=None):
        """Initialize a `MetaArray` that wraps around the input `value` and `metadata`.

        Parameters
        ----------
        value : numpy.ndarray
            The array to be wrapped.
        metadata : dict, optional
            Metadata to be wrapped, a mapping from str to str or arrays, by default None

        Raises
        ------
        ValueError
            When the input `value` is not a valid numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError("Input is not a np.ndarray.")
        self._value = value.copy()
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
    def from_hdf5(h5dt):
        """Create an instance from the content of HDF5 dataset `h5dt`.

        Parameters
        ----------
        h5dt : h5py.Dataset
            A HDF5 Dataset.

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
        dt = MetaArray(h5dt[...])
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

        def overwrite(dataset_node, input_pattern="h5_node"):
            # overwrite an existing Dataset in HDF5 file
            warn(
                f"Input `{input_pattern}` correponds to existing Dataset with name {dataset_node.name}. Overwritting..."
            )
            dataset_node[...] = self._value
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
                h5_node[name] = self._value
                for k, v in self.metadata.items():
                    h5_node[name].attrs[k] = v
        else:
            raise ValueError(
                "Input `h5_node` should be an instance of either `h5py.Dataset` or `h5py.Group`."
            )
