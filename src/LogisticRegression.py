from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from termcolor import cprint
import matplotlib.pyplot as plt

from utils import parse_args, load_yaml_config
import our_dataset as our_dataset

if __name__ == "__main__":
    subjects = [2, 4, 5, 6, 7, 10, 11]

    args = parse_args()
    config = load_yaml_config(config_filename=args.config)

    X = []
    y = []

    # Load data
    for subj in subjects:
        dataset = our_dataset.meg_dataset(config=config, s=subj, train=True)
        cprint("Subject: " + str(subj) + ", Number of samples: " + str(len(dataset)), "yellow")
        for data, label, _ in dataset:
            data = data.numpy()
            label = 0 if label == 43 else 1  # binary classification: 0 vs 1
            X.append(data)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Flatten the input for logistic regression: (N, D)
    X = X.reshape(X.shape[0], -1)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000000, C=0.05)
    clf.fit(X_train, y_train)

    # Evaluate
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Train Accuracy: {train_acc:.2f}")
    print(f"Validation Accuracy: {val_acc:.2f}")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), PCA(n_components=50), LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)
    print("Validation accuracy:", pipeline.score(X_val, y_val))

    fig, ax = plt.subplots()
    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='viridis', edgecolor='k', s=20)
    ax.set_title("PCA of Validation Set")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ## save plot
    plt.savefig("pca_validation_set.png")
    plt.show()