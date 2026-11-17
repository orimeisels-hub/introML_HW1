import numpy as np

def gensmallm(x_list: list, y_list: list, m: int):
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'
    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices][:m], y[indices][:m]

def learnknn(k: int, x_train: np.array, y_train: np.array):
    classifier = {
        'k': k,
        'x_train': x_train,
        'y_train': y_train
    }
    return classifier

def predictknn(classifier, x_test: np.array):
    k = classifier['k']
    x_train = classifier['x_train']
    y_train = classifier['y_train']
    
    y_train_flat = y_train.flatten()
    n_test = x_test.shape[0]
    predictions = np.zeros(n_test, dtype=int)
    
    # Pure matrix multiplication distance calculation - zero loops, zero memory overflow
    test_norms = np.sum(x_test**2, axis=1, keepdims=True)
    train_norms = np.sum(x_train**2, axis=1)
    dot_product = np.dot(x_test, x_train.T)
    
    distances_sq = test_norms + train_norms - 2 * dot_product
    distances_sq = np.clip(distances_sq, 0, None)
    distances = np.sqrt(distances_sq)
    
    sorted_indices = np.argsort(distances, axis=1, kind='stable')
    k_nearest_indices = sorted_indices[:, :k]
    k_nearest_labels = y_train_flat[k_nearest_indices]
    
    for i in range(n_test):
        labels = k_nearest_labels[i]
        counts = np.bincount(labels.astype(int), minlength=10)
        max_count = np.max(counts)
        best_labels = np.where(counts == max_count)[0]
        predictions[i] = np.min(best_labels)
            
    return predictions.reshape(-1, 1)

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