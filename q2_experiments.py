import numpy as np
import matplotlib.pyplot as plt
from nearest_neighbour import learnknn, predictknn, gensmallm

def run_question_2a():
    data = np.load('mnist_all.npz')
    train_lists = [data['train1'], data['train4'], data['train7'], data['train9']]
    test_lists = [data['test1'], data['test4'], data['test7'], data['test9']]
    labels = [1, 4, 7, 9]
    
    # Fast subsampled test set
    x_test_full = np.vstack([lst[:100] for lst in test_lists])
    y_test_full = np.concatenate([lbl * np.ones(100) for lbl in labels]).reshape(-1, 1)
    
    m_values = [5, 10, 20, 35, 50, 75, 100]
    n_runs = 3
    k = 1
    
    mean_errors, min_errors, max_errors = [], [], []
    print("Running experiments for Q2(a)...")
    
    for m in m_values:
        run_errors = []
        for _ in range(n_runs):
            x_train, y_train = gensmallm(train_lists, labels, m)
            classifier = learnknn(k, x_train, y_train)
            y_pred = predictknn(classifier, x_test_full)
            run_errors.append(np.mean(y_test_full != y_pred))
            
        mean_errors.append(np.mean(run_errors))
        min_errors.append(np.min(run_errors))
        max_errors.append(np.max(run_errors))
        
    plt.figure(figsize=(8, 5))
    yerr_lower = np.array(mean_errors) - np.array(min_errors)
    yerr_upper = np.array(max_errors) - np.array(mean_errors)
    
    plt.errorbar(m_values, mean_errors, yerr=[yerr_lower, yerr_upper], fmt='-o', capsize=5, color='b', label='k=1 Error')
    plt.xlabel('Training Sample Size (m)')
    plt.ylabel('Average Test Error')
    plt.title('Q2(a): Test Error vs. Training Sample Size (k=1)')
    plt.grid(True)
    plt.legend()
    plt.savefig('q2a_plot.png')
    plt.close()
    print("Graph saved: q2a_plot.png")

def run_question_2d():
    data = np.load('mnist_all.npz')
    train_lists = [data['train1'], data['train4'], data['train7'], data['train9']]
    test_lists = [data['test1'], data['test4'], data['test7'], data['test9']]
    labels = [1, 4, 7, 9]
    
    x_test_full = np.vstack([lst[:100] for lst in test_lists])
    y_test_full = np.concatenate([lbl * np.ones(100) for lbl in labels]).reshape(-1, 1)
    
    m_values = [50, 150, 300]
    k_values = range(1, 12)
    n_runs = 3
    
    plt.figure(figsize=(10, 6))
    print("Running experiments for Q2(d)...")
    
    for m in m_values:
        mean_errors_m = []
        for k in k_values:
            run_errors = []
            for _ in range(n_runs):
                x_train, y_train = gensmallm(train_lists, labels, m)
                classifier = learnknn(k, x_train, y_train)
                y_pred = predictknn(classifier, x_test_full)
                run_errors.append(np.mean(y_test_full != y_pred))
            mean_errors_m.append(np.mean(run_errors))
        
        plt.plot(list(k_values), mean_errors_m, marker='o', label=f'm = {m}')
        
    plt.xlabel('k values')
    plt.ylabel('Average Test Error')
    plt.title('Q2(d): Test Error vs k for Different Training Sample Sizes')
    plt.grid(True)
    plt.legend()
    plt.savefig('q2d_plot.png')
    plt.close()
    print("Graph saved: q2d_plot.png")

def run_question_2e():
    data = np.load('mnist_all.npz')
    train_lists = [data['train1'], data['train4'], data['train7'], data['train9']]
    test_lists = [data['test1'], data['test4'], data['test7'], data['test9']]
    labels = [1, 4, 7, 9]
    
    x_test_full = np.vstack([lst[:100] for lst in test_lists])
    y_test_full = np.concatenate([lbl * np.ones(100) for lbl in labels]).reshape(-1, 1)
    
    m_values = [50, 150, 300]
    k_values = range(1, 12)
    n_runs = 3
    
    def corrupt_labels(y_arr, rate=0.3):
        y_corr = y_arr.copy().astype(int)
        n = len(y_corr)
        indices = np.random.choice(n, int(n * rate), replace=False)
        for idx in indices:
            curr = y_corr[idx].item()
            others = [l for l in labels if l != curr]
            y_corr[idx] = np.random.choice(others)
        return y_corr

    y_test_corrupted = corrupt_labels(y_test_full, 0.3)
    
    plt.figure(figsize=(10, 6))
    print("Running experiments for Q2(e)...")
    
    for m in m_values:
        mean_errors_m = []
        for k in k_values:
            run_errors = []
            for _ in range(n_runs):
                x_train_raw, y_train_raw = gensmallm(train_lists, labels, m)
                y_train_corr = corrupt_labels(y_train_raw, 0.3)
                
                classifier = learnknn(k, x_train_raw, y_train_corr)
                y_pred = predictknn(classifier, x_test_full)
                run_errors.append(np.mean(y_test_corrupted != y_pred))
            mean_errors_m.append(np.mean(run_errors))
        
        plt.plot(list(k_values), mean_errors_m, marker='s', linestyle='--', label=f'Corrupted m = {m}')
        
    plt.xlabel('k values')
    plt.ylabel('Average Test Error (Corrupted)')
    plt.title('Q2(e): Test Error vs k with 30% Corrupted Labels')
    plt.grid(True)
    plt.legend()
    plt.savefig('q2e_plot.png')
    plt.close()
    print("Graph saved: q2e_plot.png")

if __name__ == '__main__':
    run_question_2a()
    run_question_2d()
    run_question_2e()
    print("All experiments completed successfully!")