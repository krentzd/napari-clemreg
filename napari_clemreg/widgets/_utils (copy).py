from napari.layers.base.base import Layer
from abc import abstractmethod

class Transform(Layer):

    def __init__(
        self,
        data=1,
        ndim=1
    ):

        super().__init__(
            data,
            ndim
        )

    @property
    def data(self):
        # user writes own docstring
        self.data

    @data.setter
    def data(self, data):
        self.data

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        return self.data.shape

    def _get_state(self):
        raise NotImplementedError()

    def _get_ndim(self):
        raise NotImplementedError()

    def _set_view_slice(self):
        raise NotImplementedError()

    def _update_thumbnail(self):
        raise NotImplementedError()

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : tuple
            Value of the data.
        """
        raise NotImplementedError()
