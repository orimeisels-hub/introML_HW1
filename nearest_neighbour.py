import numpy as np


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m alongside its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    return x[indices][:m], y[indices][:m]


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    # Efficiency: precompute float64 training data, centering mean, squared norms,
    # and flattened integer labels once in learnknn so repeated predictknn calls do
    # not redo this O(m * d) work.
    # Converting to float64 prevents integer overflow (e.g., uint8/int32 squaring)
    # and enables NumPy's optimized BLAS (DGEMM) matrix multiplication in predictknn.
    x_train_f = np.array(x_train, dtype=np.float64, copy=True)
    mean = np.mean(x_train_f, axis=0)
    x_train_centered = x_train_f - mean
    train_norms_sq = np.sum(x_train_centered ** 2, axis=1)
    y_train_int = np.array(y_train, dtype=int, copy=True).ravel()

    classifier = {
        'k': int(k),
        'x_train': x_train_f,
        'mean': mean,
        'x_train_centered': x_train_centered,
        'train_norms_sq': train_norms_sq,
        'y_train': y_train_int,
    }
    return classifier


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k = classifier['k']
    x_train = classifier['x_train']
    mean = classifier['mean']
    x_train_centered = classifier['x_train_centered']
    train_norms_sq = classifier['train_norms_sq']
    y_train = classifier['y_train']

    n_test = x_test.shape[0]
    m_train = x_train.shape[0]
    if n_test == 0:
        return np.empty((0, 1), dtype=int)

    # Efficiency (1) - Vectorized pairwise squared Euclidean distances via BLAS GEMM:
    # Using ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b) computes all n x m distances
    # in a single O(n * m * d) BLAS matrix multiplication with only O(n * m) memory,
    # avoiding slow Python loops and O(n * m * d) 3D broadcasting tensors.
    # Centering both x_train and x_test by the same mean preserves all Euclidean
    # distances exactly while mitigating floating-point cancellation.
    # We also omit np.sqrt because sqrt is strictly increasing on [0, inf), so
    # squared distances induce the exact same nearest-neighbor order faster.
    x_test_f = np.asarray(x_test, dtype=np.float64)
    x_test_centered = x_test_f - mean
    test_norms_sq = np.sum(x_test_centered ** 2, axis=1, keepdims=True)
    distances_sq = test_norms_sq + train_norms_sq - 2.0 * (x_test_centered @ x_train_centered.T)
    np.maximum(distances_sq, 0.0, out=distances_sq)

    # Efficiency (2) - O(n * m) top-k selection via np.argpartition instead of
    # O(n * m log m) full sorting, with exact tie-breaking fallback:
    # When no tie (or floating-point near-tie) crosses the k-th distance boundary,
    # the set of k nearest neighbors is unique and their order does not affect voting.
    # Only for rows with a tie or near-tie within floating-point rounding tolerance
    # do we recompute exact squared distances directly and run stable argsort
    # (which prefers smaller indices in x_train on ties).
    if k == m_train:
        k_nearest_labels = np.broadcast_to(y_train, (n_test, m_train))
    else:
        k_nearest_indices = np.argpartition(distances_sq, k - 1, axis=1)[:, :k]
        kth_vals = np.max(
            np.take_along_axis(distances_sq, k_nearest_indices, axis=1),
            axis=1,
            keepdims=True,
        )
        scale = test_norms_sq + np.max(train_norms_sq)
        tol = 1e-11 * scale + 1e-12
        tie_rows = np.where(np.sum(distances_sq <= kth_vals + tol, axis=1) > k)[0]
        for r in tie_rows:
            exact_d_sq = np.sum((x_test_f[r] - x_train) ** 2, axis=1)
            k_nearest_indices[r] = np.argsort(exact_d_sq, kind='stable')[:k]
        k_nearest_labels = y_train[k_nearest_indices]

    # Efficiency (3) - Fully vectorized majority voting (no Python loop over n_test):
    # Labels are integers in {0, ..., 9}, so we count votes across all test examples
    # simultaneously in NumPy. np.argmax returns the first (smallest) label achieving
    # the maximum count, automatically satisfying the majority-vote tie-breaking rule.
    counts = (k_nearest_labels[:, :, None] == np.arange(10)).sum(axis=1)
    predictions = np.argmax(counts, axis=1).reshape(-1, 1)

    return predictions


def simple_test():
    data = np.load('mnist_all.npz')
    train1, train4, train7, train9 = data['train1'], data['train4'], data['train7'], data['train9']
    test1, test4, test7, test9 = data['test1'], data['test4'], data['test7'], data['test9']

    x_train, y_train = gensmallm([train1, train4, train7, train9], [1, 4, 7, 9], 100)
    x_test, y_test = gensmallm([test1, test4, test7, test9], [1, 4, 7, 9], 50)

    classifer = learnknn(5, x_train, y_train)
    preds = predictknn(classifer, x_test)

    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    i = np.random.randint(0, x_test.shape[0])
    print(f"The {i}'th test sample was classified as {preds[i]}")


if __name__ == '__main__':
    simple_test()
