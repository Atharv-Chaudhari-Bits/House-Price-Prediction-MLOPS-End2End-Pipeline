import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_and_save_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    df.to_csv("data/raw/housing.csv", index=False)
    print("Data saved to data/raw/housing.csv")

if __name__ == "__main__":
    load_and_save_data()