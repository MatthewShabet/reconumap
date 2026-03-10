# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import locale

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin
from sklearn.utils import check_array

import numpy as np
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
from scipy.spatial import ConvexHull
import numba

from .layouts import optimize_layout_euclidean

locale.setlocale(locale.LC_NUMERIC, "C")


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAP(BaseEstimator, ClassNamePrefixFeaturesOutMixin):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    init: string
        A numpy array of initial embedding positions
    
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    set_op_mix_ratio: float (optional, default 1.0) [fss]

    local_connectivity: int (optional, default 1) [fss]

    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    tqdm_kwds: dict (optional, defaul None)
        Key word arguments to be used by the tqdm progress bar.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """

    def __init__(
        self,
        init,
        n_epochs=None,
        min_dist=0.1,
        spread=1.0,
        a=None,
        b=None,
        repulsion_strength=1.0,
        learning_rate=1.0,
        negative_sample_rate=5,
        tqdm_kwds=None,
        
        area=10.0,

        n_neighbors=15,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        
        n_jobs=-1,
    ):
        self.init = init
        self.n_epochs = n_epochs ; assert self.n_epochs > 10 and isinstance(self.n_epochs, int)
        self.min_dist = min_dist ; assert self.min_dist >= 0.0
        self.spread = spread     ; assert self.spread >= self.min_dist
        self.a = a
        self.b = b
        self.repulsion_strength = repulsion_strength     ; assert self.repulsion_strength >= 0
        self.learning_rate = learning_rate               ; assert self.learning_rate >= 0
        self.negative_sample_rate = negative_sample_rate ; assert self.negative_sample_rate >= 0

        self.area = area ; assert self.area > 0
        
        self.n_neighbors = n_neighbors     ; assert self.n_neighbors >= 2
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        
        self.n_jobs = n_jobs               ; assert self.n_jobs == -1 or self.n_jobs > 0
        
        self.tqdm_kwds = tqdm_kwds
        if self.tqdm_kwds is None:
            self.tqdm_kwds = {}
        assert isinstance(self.tqdm_kwds, dict)
        if "desc" not in self.tqdm_kwds:
            self.tqdm_kwds["desc"] = "Epochs completed"
        if "bar_format" not in self.tqdm_kwds:
            bar_f = "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]"
            self.tqdm_kwds["bar_format"] = bar_f


    def fit(self, X, **kwargs):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_samples)
            X must be a square distance matrix
        """
        X = check_array(
            X,
            dtype=np.float32,
            accept_sparse="csr",
            order="C",
            ensure_all_finite=True,
        )
        assert scipy.sparse.isspmatrix_csr(X)
        assert X.shape[0] == X.shape[1]
        if not X.has_sorted_indices:
            X.sort_indices()
        assert X.shape[0] > self.n_neighbors, "n_neighbors is larger than the dataset size"
        assert sparse_tril(X).getnnz() == sparse_triu(X).getnnz()
        assert np.all(X.diagonal() == 0)

        assert isinstance(self.init, np.ndarray)
        init = check_array(
            self.init,
            dtype=np.float32,
            accept_sparse=False,
            ensure_all_finite=True,
        )

        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        self._original_n_threads = numba.get_num_threads()
        if self.n_jobs > 0 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)

        ################################################################################
        print("Construct fuzzy simplicial set")

        from .fss import fuzzy_simplicial_set
        self.graph_ = fuzzy_simplicial_set(
            X,
            self.n_neighbors,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
        )
        
        # Assert the number of vertices with degree 0 in our our umap.graph_
        assert np.sum(np.array(self.graph_.sum(axis=0)).flatten() == 0) == 0
        assert np.sum(np.array(self.graph_.sum(axis=1)).flatten() == 0) == 0
        
        assert isinstance(self.graph_, scipy.sparse._csr.csr_matrix), f"{type(self.graph_)}"
        assert isinstance(X, scipy.sparse._csr.csr_matrix), f"{type(self.graph_)}"
        assert X.shape == self.graph_.shape
        #assert X.nnz == self.graph_.nnz
        #assert np.array_equal(X.indptr, self.graph_.indptr)
        #assert np.array_equal(X.indices, self.graph_.indices)

        ################################################################################
        print("Construct embedding")
        """Perform a fuzzy simplicial set embedding, using a specified
        initialisation method and then minimizing the fuzzy set cross entropy
        between the 1-skeletons of the high and low dimensional fuzzy simplicial
        sets."""

        # The 1-skeleton of the high dimensional fuzzy simplicial set as represented by a graph for which we require a sparse matrix for the (weighted) adjacency matrix.
        graph = self.graph_
        graph = graph.tocoo()
        graph.sum_duplicates()
        assert np.all(graph.data >= 0)
        graph.data[graph.data < (graph.data.max() / float(self.n_epochs))] = 0.0
        graph.eliminate_zeros()
        assert graph.shape[0] == graph.shape[1] == init.shape[0]

        # make_epochs_per_sample
        epochs_per_sample = np.float64(graph.data.max()) / np.float64(graph.data)
        
        # How to initialize the low dimensional embedding. A numpy array of initial embedding positions.
        init = np.array(init)
        print(f"Duplicate initializations: {init.shape[0] - np.unique(init, axis=0).shape[0]}/{init.shape[0]}")
        embedding = init
        
        print(f"Original volume: {ConvexHull(embedding).volume}")
        embedding = embedding - embedding.mean(axis=0)
        embedding = (
            np.sqrt(self.area) * embedding / np.sqrt(ConvexHull(embedding).volume)
        ).astype(np.float32, order="C")
        print(f"Init volume: {ConvexHull(embedding).volume}")
        
        embedding = optimize_layout_euclidean(
            head_embedding=embedding,
            tail_embedding=embedding,
            head=graph.row,
            tail=graph.col,
            #weight=graph.data,
            n_epochs=self.n_epochs,
            n_vertices=graph.shape[1],
            epochs_per_sample=epochs_per_sample,
            a=self._a,
            b=self._b,
            gamma=self.repulsion_strength,
            initial_alpha=self.learning_rate,
            negative_sample_rate=self.negative_sample_rate,
            tqdm_kwds=self.tqdm_kwds,
            move_other=True,
        )
        
        print("Finished embedding")
        print(f"Final volume: {ConvexHull(embedding).volume}")

        numba.set_num_threads(self._original_n_threads)

        return embedding
