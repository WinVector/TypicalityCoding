
import numpy as np
from numpy.linalg import norm, svd


class SpheringTransform:
    """
    Full rank sphering or whitening transform (ref: https://en.wikipedia.org/wiki/Whitening_transformation ).
    Full rank affine transform such that E[x.x] = 1 on row of training data.
    """
    def __init__(self, *, epsilon:float = 1e-6) -> None:
        """
        :param epsilon: sphering parameter, assigns how much stretch is allowed in previously constant dimensions. protects against degenerate data.
        """
        epsilon = float(epsilon)
        assert epsilon > 0
        self.epsilon = epsilon
        self.scale = 1
        self.center = None
        self.xform = None
    
    def fit(self, X) -> "SpheringTransform":
        """
        Build a sphering transform on row-oriented data such that E[x.x] = 1 on rows of training data.
        Ref: https://en.wikipedia.org/wiki/Whitening_transformation .
        This transform is picked to be full rank (not dimension reducing), so can measure unexpected variation in new data.
        The sphering transform is row_x -> scale * (row_x - center) @ xform. This transform is implemented by transform().

        :param X: data set, rows are instances, columns are numeric values. Has more rows than columns, and has non-zero data.
        :return: self (for method chaining)
        """
        # clear out any previous result
        self.scale = 1
        self.center = None
        self.xform = None
        # convert data to numpy array to avoid later problems
        d_centered = np.asarray(X)
        # get center in original space
        center = np.mean(d_centered, axis=0)
        # center data around origin
        d_centered = d_centered - center
        # compute inertial ellipsoid around origin
        inertial = np.matmul(d_centered.transpose(), d_centered)
        del d_centered
        # eliminate semi-definite case
        inertial = inertial + self.epsilon * np.identity(inertial.shape[0])
        # work out a transformation so inertial ellipsoid would have been a sphere
        # (this is factoring the inverse of inertial using SVD)
        svd_result = svd(inertial, hermitian=True)
        # don't allow too small singular values when converting to full rank
        s_bounded = np.maximum(svd_result.S, self.epsilon)
        xform = svd_result.U * np.sqrt(1 / s_bounded)
        # save result as attributes
        self.center = center
        self.xform = xform
        # optional rescale so mean square norm is 1 on training data
        x_transformed = self.transform(X)
        x_norm_sq = norm(x_transformed, axis=1)**2
        self.scale = 1/np.sqrt(np.mean(x_norm_sq))
        return self

    def transform(self, X) -> np.ndarray:
        """
        Apply sphering transform to new data.

        :param X: data
        :return: transformed data
        """
        X = np.asarray(X)
        # apply centering and spherizing transform to new data
        v = ((X - self.center) @ self.xform)
        v = self.scale * v
        return v
