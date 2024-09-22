import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def pair_plot(df, hue_column):
    plt.figure(figsize=(20, 15))
    sns.pairplot(df, hue=hue_column, height=2.5, plot_kws={'s':50, 'alpha': 0.7})
    plt.tight_layout(pad=3)
    plt.show()

def main():
    df = pd.read_csv('C:/Users/said/Desktop/choixpeau/datasets/dataset_train.csv')
    df_num = df.select_dtypes(include='number')
    df_with_house_name = pd.concat([df_num, df['Hogwarts House']], axis=1)
    pair_plot(df_with_house_name, hue_column='Hogwarts House')

if __name__ == "__main__":
    main()