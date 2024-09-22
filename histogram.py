import matplotlib.pyplot as plt
import pandas as pd

def histogram(group_data, subjects):
    subject_len = len(subjects)
    num_col = 4
    num_row = ((subject_len + num_col - 1) // num_col)
    fig, axes = plt.subplots(num_row, num_col, figsize=(15, num_row * 5))
    axes = axes.flatten()
    for i, subject in enumerate(subjects):
        ax = axes[i]
        for house, group in group_data:
            ax.hist(group[subject].dropna(), bins=5, alpha=0.4, label=house)
        ax.set_xlabel(f"{subject}")
        ax.set_ylabel("students")
    for j in range(subject_len, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(pad=5)
    plt.show()

def main():
    df = pd.read_csv('C:/Users/said/Desktop/choixpeau/datasets/dataset_train.csv')
    group_data = df.groupby('Hogwarts House')
    subjects = subjects = df.columns[6:].tolist()
    histogram(group_data, subjects)

if __name__ == "__main__":
    main()