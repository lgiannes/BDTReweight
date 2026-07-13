import numpy as np
from hep_ml import reweight
import pickle
from numpy.typing import ArrayLike


class Reweighter(reweight.GBReweighter):
    """
    A reweighter class inherited from hep_ml.reweight.GBReweighter,
    with extended functions to predict weights and save to / load
    from pickle object.
    
    """
    
    def __init__(self, n_estimators=40, learning_rate=0.2, max_depth=3, min_samples_leaf=200, loss_regularization=5, gb_args=None):
        super().__init__(n_estimators, learning_rate, max_depth, min_samples_leaf, loss_regularization, gb_args)
        self.xsec_scale_factor = 1.0  # Cross-section scale factor (e.g., sigma_target / sigma_source)
        self.norm_factor = 1.0  # Normalization factor to preserve total weights

    def predict_matched_total_weights(self, original : np.ndarray, original_weight : ArrayLike = None, target_weight : ArrayLike = None) -> ArrayLike:
        """
        hep_ml.reweight's GBReweighter.predict_weights() doesn't
        preserve the total weights after reweight. In this modified
        version, the total weights are  either preserved, or matched
        to target total weights. The target weights normalization already
        accounts for any cross-section scaling (xsec_scale_factor).

        Parameters
        ----------
        original : np.ndarray
            The source sample arrays of neutrino MC variables. 
        original_weight : ArrayLike, optional
            The old weights of source sample events.
        target_weight : ArrayLike, optional
            The weights or target sample events. If provided,
            source sample predicted weights' total magnitude
            will be matched to sum(target_weight).

        Returns
        ----------
        ArrayLike
        """
        new_weights = self.predict_weights(original, original_weight=original_weight)
        if target_weight is None:
            # Ensure sum(new_weights) = len(original) 
            new_weights = new_weights * len(new_weights)/np.sum(new_weights)
        else:
            new_weights = new_weights * ( np.sum(target_weight)/(np.sum(new_weights)) )
        return new_weights

    def set_weight_normalization_factor(self, original : np.ndarray, original_weight : ArrayLike = None) -> float:
        """
        Compute the normalization factor to be applied to predicted weights
        to match the total weights of the target sample.

        Parameters
        ----------
        original : np.ndarray
            The source sample arrays of neutrino MC variables.
        original_weight : ArrayLike, optional
            The old weights of source sample events.

        Returns
        ----------
        float
            The normalization factor to apply to predicted weights.
        """
        new_weights = self.predict_weights(original, original_weight=original_weight)
        self.norm_factor = len(new_weights)/np.sum(new_weights)

    def predict_weight_single_event(self, features : ArrayLike, verbose : bool = False) -> float:
        """
        Predict weights for a single event given its features.

        Parameters
        ----------
        features : ArrayLike
            The source sample features of neutrino MC variables
            (values of reweight variables).
        verbose : bool, optional
            If True, print the intermediate factors that make up the
            returned weight.

        Returns
        ----------
        float
        """
        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
        w = self.predict_weights(X)

        if verbose:
            print(f"SUPERVERBOSE-- w={w[0]} * {self.xsec_scale_factor} * {self.norm_factor} = {w[0] * self.xsec_scale_factor * self.norm_factor}")

        return w[0] * self.xsec_scale_factor * self.norm_factor

    def set_xsec_scale_factor(self, scale_factor: float):
        """
        Set the cross-section scale factor to be applied to all predicted weights.
        This allows the pickle file to be self-contained with the total cross-section weight.

        Parameters
        ----------
        scale_factor : float
            The scale factor (typically sigma_target / sigma_source )

        Returns
        ----------
        None
        """
        self.xsec_scale_factor = scale_factor




    def save_to_pickle(self, filepath : str):
        """
        Save Reweighter object via pickle.

        Parameters
        ----------
        filepath : str
            The file path to save to.
 
        Returns
        ----------
        None
        """
        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(filepath):
        """
        Load Reweighter object via pickle.

        Parameters
        ----------
        filepath : str
            The file path to load from.
 
        Returns
        ----------
        Reweighter
        """
        with open(filepath, 'rb') as input:
            reweighter = pickle.load(input)

        return reweighter