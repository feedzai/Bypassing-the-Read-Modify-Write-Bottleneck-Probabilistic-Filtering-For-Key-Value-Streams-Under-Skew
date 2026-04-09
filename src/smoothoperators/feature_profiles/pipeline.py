"""Module with the pipeline class."""
from typing import Callable, Dict, List, Optional


class Pipeline:
    """Pipeline class. This class encompasses a list of
    transformations that can be applied to data.


    Example usage:

    >>> from smoothoperators import feature_profiles
    >>> pipeline = feature_profiles.Pipeline()
    >>> pipeline.add_transformation(transformation, {'arg_1': 'value_1'})
    >>> pipeline.run()

    """

    def __init__(self):
        self.transformations = []

    def add_transformation(self,
                           transformation: Callable,
                           args: Dict = None,
                           name: Optional[str] = None) -> None:
        """Adds a transformation to the pipeline

        Args:
          transformation: Function that applies a transformation inside a
          pipeline.
          args: Arguments to be passed to `transformation`

        """
        if args is None:
            args = {}
        self.transformations.append(
            (transformation, args, name if name is not None else
             f'transformation_{len(self.transformations)}'))

    def run(self, capture_results: bool = False, log_results: bool = False):
        """Runs the pipeline

        Args:
          capture_results: If set to true we store the intermediate results
          returned by the transformation functions.

        """
        results = []
        for f, args, name in self.transformations:
            if log_results:
                print(f'Running transformation: {name}')
            r = f(**args)
            if capture_results:
                results.append(r)
            if log_results:
                print(r)
        return results
