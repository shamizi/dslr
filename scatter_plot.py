import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def scatterplot(df, correlation_matrix):
    plt.figure(figsize=(15,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap de correlation')
    plt.show()

    # Scatter plot pour Defense Against the Dark Arts et Astronomy
    plt.scatter(df['Defense Against the Dark Arts'], df['Astronomy'])
    plt.xlabel('Defense contre les force du mal')
    plt.ylabel('astronomy')
    plt.show()

def main():
    df = pd.read_csv('C:/Users/said/Desktop/choixpeau/datasets/dataset_train.csv')
    df_num = df.select_dtypes(include='number')
    correlation_matrix = df_num.corr()
    scatterplot(df_num, correlation_matrix)

if __name__ == "__main__":
    main()