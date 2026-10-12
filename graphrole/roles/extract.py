from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from graphrole.roles.description_length import get_description_length_costs
from graphrole.roles.factor import encode, get_nmf_decomposition
from graphrole.types import DataFrameLike, FactorTuple, Node


class RoleExtractor:

    """ Assign node roles based on input features """

    N_ROLE_RANGE = (2, 8)
    N_BIT_RANGE = (1, 8)

    def __init__(
        self,
        n_roles: Optional[int] = None,
        n_role_range: Optional[Tuple[int, int]] = None,
        n_bit_range: Optional[Tuple[int, int]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        :param n_roles: optional number of roles to select; default uses MDL model selection
        :param n_role_range: optional tuple for (min, max) roles for model selection grid search
        :param n_bit_range: optional tuple for (min, max) bits for model selection grid search
        :param random_state: optional seed for the NMF initialization (reproducible factors)
        """
        self.n_roles = n_roles

        self.min_roles, self.max_roles = n_role_range if n_role_range else self.N_ROLE_RANGE
        self.min_bits, self.max_bits = n_bit_range if n_bit_range else self.N_BIT_RANGE
        self.random_state = random_state

        # NMF factors (as in RolX: quantization is only used to compute description
        # lengths for model selection, not to represent the roles themselves)
        self.node_role_factor: Optional[pd.DataFrame] = None
        self.role_feature_factor: Optional[pd.DataFrame] = None
        # quantized factors of the selected model, as scored by MDL
        self.node_role_factor_encoded: Optional[pd.DataFrame] = None
        self.role_feature_factor_encoded: Optional[pd.DataFrame] = None
        self.n_bits: Optional[int] = None

    @property
    def roles(self) -> Optional[Dict[Node, float]]:
        """
        Return dict mapping node to role
        """
        try:
            role_df = self.node_role_factor.idxmax(axis=1)
            return role_df.to_dict()
        except AttributeError:
            return None

    @property
    def role_percentage(self) -> Optional[DataFrameLike]:
        """
        Return DataFrameLike with percent role (columns) for each node (index)
        """
        try:
            return self.node_role_factor.apply(lambda row: row / row.sum(), axis=1)
        except AttributeError:
            return None

    def extract_role_factors(
        self,
        features: pd.DataFrame,
    ) -> None:
        """
        Wrapper for extracting role factors from a node feature DataFrame
        and returning factor DataFrames
        :param features: DataFrame with rows of node features
        """
        if self.n_roles:
            # factors will be of shape (n_nodes x n_roles) and (n_roles x n_features) for
            # a total of n_roles * (n_nodes + n_features), so encode with approximately
            # log2(n_roles * (n_nodes + n_features)) bits
            n_bits = int(np.log2(self.n_roles * min(features.shape)))
            model = get_nmf_decomposition(features.values, self.n_roles, self.random_state)
            model_encoded = self._encode_factors(model, n_bits)
        else:
            model, model_encoded, n_bits = self._select_model(features)

        role_labels = [
            f'role_{i}'
            for i in range(model[0].shape[1])
        ]

        self.node_role_factor, self.role_feature_factor = self._factors_to_frames(
            model, features, role_labels
        )
        self.node_role_factor_encoded, self.role_feature_factor_encoded = self._factors_to_frames(
            model_encoded, features, role_labels
        )
        self.n_bits = n_bits

    def explain(self):
        raise NotImplementedError('Role explanation (\"sense making\") is not yet implemented.')

    @staticmethod
    def _factors_to_frames(
        model: FactorTuple,
        features: pd.DataFrame,
        role_labels: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Wrap (node x role, role x feature) factor ndarrays as labelled DataFrames
        """
        G, F = model
        return (
            pd.DataFrame(G, index=features.index, columns=role_labels),
            pd.DataFrame(F, index=role_labels, columns=features.columns),
        )

    def _select_model(
        self,
        features: pd.DataFrame,
    ) -> Tuple[FactorTuple, FactorTuple, int]:
        """
        Select optimal model via grid search over n_roles and n_bits as measured
        by Minimum Description Length; returns the NMF factors of the selected
        model, their quantized (encoded) version, and the selected n_bits
        :param features: DataFrame with rows of node features
        """
        # define grid
        max_bits_grid_idx = self.max_bits + 1
        max_roles_grid_idx = min(min(features.shape), self.max_roles) + 1
        n_bits_grid = range(self.min_bits, max_bits_grid_idx)
        n_roles_grid = range(self.min_roles, max_roles_grid_idx)

        # ndarrays to store encoding and error costs
        matrix_dims = (max_roles_grid_idx, max_bits_grid_idx)
        encoding_costs = np.full(matrix_dims, np.nan)
        error_costs = np.full(matrix_dims, np.nan)
        # dicts to store NMF factor tuples (one per n_roles) and encoded factor tuples
        factors = {}
        encoded_factors = defaultdict(dict)

        # grid search
        for roles in n_roles_grid:
            # the NMF does not depend on the number of bits, so factor once per n_roles
            factors[roles] = get_nmf_decomposition(features.values, roles, self.random_state)
            for bits in n_bits_grid:

                try:
                    model = self._encode_factors(factors[roles], bits)
                    encoding_cost, error_cost = get_description_length_costs(features, model)
                except ValueError:
                    # raised when bits is too large to quantize the number of samples
                    continue

                encoding_costs[roles, bits] = encoding_cost
                error_costs[roles, bits] = error_cost
                encoded_factors[roles][bits] = model

        # select factors with minimal cost
        costs = self._rescale_costs(encoding_costs) + self._rescale_costs(error_costs)
        min_cost = np.nanmin(costs)
        # we could catch an IndexError here, but if np.argwhere returns empty there is
        # no way to handle model selection and hence no way to recover
        min_role, min_bits = np.argwhere(costs == min_cost)[0]
        return factors[min_role], encoded_factors[min_role][min_bits], int(min_bits)

    @staticmethod
    def _encode_factors(
        model: FactorTuple,
        n_bits: int,
    ) -> FactorTuple:
        """
        Encode (quantize) NMF factor matrices
        :param model: tuple of (node x role, role x feature) factor matrices
        :param n_bits: number of bits to use for encoding factor matrices
        """
        n_bins = int(2**n_bits)
        G, F = model
        return encode(G, n_bins), encode(F, n_bins)

    @staticmethod
    def _get_encoded_role_factors(
        features: pd.DataFrame,
        n_roles: int,
        n_bits: int,
    ) -> FactorTuple:
        """
        Compute encoded NMF decomposition of feature DataFrame
        :param features: DataFrame with rows of node features
        :param n_roles: number of roles (rank of NMF decomposition)
        :param n_bits: number of bits to use for encoding factor matrices
        """
        model = get_nmf_decomposition(features.values, n_roles)
        return RoleExtractor._encode_factors(model, n_bits)

    @staticmethod
    def _rescale_costs(
        costs: np.ndarray
    ) -> np.ndarray:
        """
        Rescale the cost matrices for a fixed n_role so that
        encoding and error costs are on the same scale
        :param costs: matrix of costs with n_roles across roles and n_bits across columns
        """
        norms_skip_nan = np.sqrt(np.nansum(np.square(costs), axis=1))
        return costs / norms_skip_nan.reshape(costs.shape[0], 1)
