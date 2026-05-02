import pandas as pd


def main():
    df = pd.read_csv("TMDB_movie_dataset_v11.csv")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Null counts:\n", df.isnull().sum())
    df.info()

    drop_cols = [
        "id",
        "vote_average",
        "homepage",
        "poster_path",
        "backdrop_path",
        "production_companies",
        "production_countries",
        "spoken_languages",
    ]
    
    df_cleaned = df.drop(columns=drop_cols)
    df_cleaned = df_cleaned.dropna()
    print("Cleaned shape:", df_cleaned.shape)
    print("Cleaned columns:", df_cleaned.columns.tolist())
    print("Cleaned null counts:\n", df_cleaned.isnull().sum())
    df_cleaned.info()
    
    df_cleaned.to_csv("cleaned_dataset.csv", index=False)


if __name__ == "__main__":
    main()