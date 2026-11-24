import numpy as np
import matplotlib.pyplot as plt
from nearest_neighbour import learnknn, predictknn, gensmallm


LABELS = [1, 4, 7, 9]


def load_mnist_subset(path: str = 'mnist_all.npz'):
    """Load the digits 1, 4, 7, 9 from mnist_all.npz once (full training & test sets)."""
    data = np.load(path)
    train_lists = [data[f'train{d}'].astype(np.float64) for d in LABELS]
    test_lists = [data[f'test{d}'].astype(np.float64) for d in LABELS]

    # Full test set (all 4,154 images across digits 1, 4, 7, 9)
    x_test_full = np.vstack(test_lists)
    y_test_full = np.concatenate(
        [lbl * np.ones(len(lst), dtype=int) for lbl, lst in zip(LABELS, test_lists)]
    ).reshape(-1, 1)

    return train_lists, x_test_full, y_test_full


def corrupt_labels(y_arr: np.ndarray, labels: list = LABELS, rate: float = 0.3) -> np.ndarray:
    """
    Select a random `rate` fraction of examples and change each of their labels
    uniformly at random to one of the other possible labels (fully vectorized).
    """
    y_corr = np.array(y_arr, dtype=int, copy=True, order='C')
    flat = y_corr.ravel()
    n = flat.shape[0]
    n_corrupt = int(round(n * rate))
    corrupt_idx = np.random.choice(n, size=n_corrupt, replace=False)

    labels_arr = np.array(labels, dtype=int)
    label_to_pos = np.empty(10, dtype=int)
    label_to_pos[labels_arr] = np.arange(len(labels_arr))

    curr_pos = label_to_pos[flat[corrupt_idx]]
    offsets = np.random.randint(1, len(labels_arr), size=n_corrupt)
    flat[corrupt_idx] = labels_arr[(curr_pos + offsets) % len(labels_arr)]

    return y_corr


def run_question_2a(train_lists, x_test_full, y_test_full):
    """Q2(a): k=1, m in [1, 100], 10 runs per m on the full test sample with min/max error bars."""
    m_values = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_runs = 10
    k = 1

    mean_errors, min_errors, max_errors = [], [], []
    print(f"Running Q2(a) on full test set ({len(x_test_full)} samples), {n_runs} runs per m...")

    for m in m_values:
        run_errors = []
        for _ in range(n_runs):
            x_train, y_train = gensmallm(train_lists, LABELS, m)
            classifier = learnknn(k, x_train, y_train)
            y_pred = predictknn(classifier, x_test_full)
            run_errors.append(np.mean(y_test_full != y_pred))

        mean_err = float(np.mean(run_errors))
        min_err = float(np.min(run_errors))
        max_err = float(np.max(run_errors))
        mean_errors.append(mean_err)
        min_errors.append(min_err)
        max_errors.append(max_err)
        print(
            f"  m={m:3d}: mean={mean_err:.4f}, min={min_err:.4f}, "
            f"max={max_err:.4f}, range={max_err - min_err:.4f}"
        )

    plt.figure(figsize=(9, 5.5))
    yerr_lower = np.array(mean_errors) - np.array(min_errors)
    yerr_upper = np.array(max_errors) - np.array(mean_errors)

    plt.errorbar(
        m_values,
        mean_errors,
        yerr=[yerr_lower, yerr_upper],
        fmt='-o',
        capsize=5,
        color='tab:blue',
        label='Average test error (bars: min–max over 10 runs)',
    )
    plt.xlabel('Training Sample Size (m)')
    plt.ylabel('Average Test Error')
    plt.title('Q2(a): 1-NN Average Test Error vs. Training Sample Size (10 runs)')
    plt.ylim(0, max(max_errors) * 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q2a_plot.png', dpi=150)
    plt.close()
    print("Saved: q2a_plot.png\n")


def run_question_2d(train_lists, x_test_full, y_test_full):
    """Q2(d): m in {50, 150, 500}, k in 1..15, 30 runs per m, separate plot for each m."""
    m_values = [50, 150, 500]
    k_values = list(range(1, 16))
    n_runs = 30

    print(f"Running Q2(d) on full test set ({len(x_test_full)} samples), {n_runs} runs per m...")
    results = {}

    for m in m_values:
        errors = np.zeros((n_runs, len(k_values)), dtype=np.float64)
        for r in range(n_runs):
            x_train, y_train = gensmallm(train_lists, LABELS, m)
            for idx_k, k in enumerate(k_values):
                classifier = learnknn(k, x_train, y_train)
                y_pred = predictknn(classifier, x_test_full)
                errors[r, idx_k] = np.mean(y_test_full != y_pred)

        mean_errors_m = errors.mean(axis=0)
        results[m] = mean_errors_m
        best_idx = int(np.argmin(mean_errors_m))
        best_k = k_values[best_idx]
        best_err = mean_errors_m[best_idx]
        print(f"  m={m:3d}: optimal k = {best_k} (error = {best_err:.4f})")
        print(
            "    errors by k: "
            + ", ".join(f"k={k}:{e:.4f}" for k, e in zip(k_values, mean_errors_m))
        )

        plt.figure(figsize=(8.5, 5))
        plt.plot(k_values, mean_errors_m, marker='o', color='tab:blue', label=f'm = {m}')
        plt.scatter([best_k], [best_err], color='tab:red', zorder=5, s=80, label=f'Optimal k = {best_k} ({best_err:.4f})')
        plt.xticks(k_values)
        plt.xlabel('k (number of nearest neighbors)')
        plt.ylabel('Average Test Error')
        plt.title(f'Q2(d): Test Error vs. k for m = {m} (Clean Labels, 30 runs)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        filename = f'q2d_m{m}_plot.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    print()
    return results


def run_question_2e(train_lists, x_test_full, y_test_full):
    """Q2(e): Same as Q2(d), but corrupt 30% of labels in every training and test set."""
    m_values = [50, 150, 500]
    k_values = list(range(1, 16))
    n_runs = 30

    print(f"Running Q2(e) (30% corrupted labels) on full test set ({len(x_test_full)} samples), {n_runs} runs per m...")
    results = {}

    for m in m_values:
        errors = np.zeros((n_runs, len(k_values)), dtype=np.float64)
        for r in range(n_runs):
            x_train, y_train_raw = gensmallm(train_lists, LABELS, m)
            y_train_corr = corrupt_labels(y_train_raw, LABELS, rate=0.3)
            y_test_corr = corrupt_labels(y_test_full, LABELS, rate=0.3)

            for idx_k, k in enumerate(k_values):
                classifier = learnknn(k, x_train, y_train_corr)
                y_pred = predictknn(classifier, x_test_full)
                errors[r, idx_k] = np.mean(y_test_corr != y_pred)

        mean_errors_m = errors.mean(axis=0)
        results[m] = mean_errors_m
        best_idx = int(np.argmin(mean_errors_m))
        best_k = k_values[best_idx]
        best_err = mean_errors_m[best_idx]
        print(f"  m={m:3d}: optimal k = {best_k} (error = {best_err:.4f})")
        print(
            "    errors by k: "
            + ", ".join(f"k={k}:{e:.4f}" for k, e in zip(k_values, mean_errors_m))
        )

        plt.figure(figsize=(8.5, 5))
        plt.plot(k_values, mean_errors_m, marker='s', color='tab:orange', label=f'Corrupted m = {m}')
        plt.scatter([best_k], [best_err], color='tab:red', zorder=5, s=80, label=f'Optimal k = {best_k} ({best_err:.4f})')
        plt.xticks(k_values)
        plt.xlabel('k (number of nearest neighbors)')
        plt.ylabel('Average Test Error')
        plt.title(f'Q2(e): Test Error vs. k for m = {m} (30% Corrupted Labels, 30 runs)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        filename = f'q2e_m{m}_plot.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    print()
    return results


if __name__ == '__main__':
    np.random.seed(42)
    train_lists, x_test_full, y_test_full = load_mnist_subset('mnist_all.npz')
    run_question_2a(train_lists, x_test_full, y_test_full)
    run_question_2d(train_lists, x_test_full, y_test_full)
    run_question_2e(train_lists, x_test_full, y_test_full)
    print("All experiments completed successfully!")
