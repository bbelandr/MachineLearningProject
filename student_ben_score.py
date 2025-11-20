import os
import polars as pl # Using polars instead of pandas for speed. >9 million lines in 784k csv files.

# ben_score
# Resets for every new student
# A running total. Adds 1 for a correct answer, subtracts 1 for a wrong answer.

def calculate_ben_score(df):
    """Calculate ben_score: running total of +1 for correct, -1 for incorrect, resets per student"""
    print("Sorting data by student_id and timestamp...")
    df = df.sort(["student_id", "timestamp"])
    
    print("Detecting student changes...")
    # Detect when student ID changes
    df = df.with_columns(
        pl.when(pl.col('student_id') != pl.col('student_id').shift(1))
        .then(1)
        .otherwise(0)
        .alias('student_changed')
    )
    
    print("Creating student groups...")
    # Create groups for each student
    df = df.with_columns(
        pl.col('student_changed').cum_sum().alias('student_group_id')
    )
    
    print("Computing ben_score...")
    # Calculate ben_score: running total within each student group
    # +1 for correct (1), -1 for incorrect (0)
    df = df.with_columns(
        (pl.when(pl.col('correct') == 1).then(1).otherwise(-1)).cum_sum().over('student_group_id').alias('student_ben_score')
    )
    
    print("Cleaning up temporary columns...")
    # Drop temporary columns
    df = df.drop(['student_changed', 'student_group_id'])
    
    return df


def calculate_forgetful_ben_score(df, history):
    """Calculate ben_score based only on the previous N answers within each student"""
    print("Sorting data by student_id and timestamp...")
    df = df.sort(["student_id", "timestamp"])
    
    print("Detecting student changes...")
    # Detect when student ID changes
    df = df.with_columns(
        pl.when(pl.col('student_id') != pl.col('student_id').shift(1))
        .then(1)
        .otherwise(0)
        .alias('student_changed')
    )
    
    print("Creating student groups...")
    # Create groups for each student
    df = df.with_columns(
        pl.col('student_changed').cum_sum().alias('student_group_id')
    )
    
    print(f"Computing forgetful ben_score (window of {history})...")
    # Calculate forgetful ben_score: rolling sum of previous N answers within each student
    # +1 for correct (1), -1 for incorrect (0)
    # Fill nulls with 0 so early answers start at 0 and build incrementally
    df = df.with_columns(
        (pl.when(pl.col('correct') == 1).then(1).otherwise(-1))
        .rolling_sum(window_size=history, min_periods=1)
        .over('student_group_id')
        .fill_null(0)
        .alias(f'forgetful_ben_score={history}')
    )
    
    print("Cleaning up temporary columns...")
    # Drop temporary columns
    df = df.drop(['student_changed', 'student_group_id'])
    
    return df

# We split our data into 2 separate sets in the prep_data file
print("Loading training data...")
train_df = pl.read_parquet(rf".\Data\final_train_data.parquet")
print("Loading validation data...")
val_df = pl.read_parquet(rf".\Data\final_val_data.parquet")

print("\nCalculating scores for training data...")
train_df = calculate_ben_score(train_df)
print("\nCalculating scores for validation data...")
val_df = calculate_ben_score(val_df)
val_df = calculate_forgetful_ben_score(val_df, 5)

print("\nTraining data sample:")
print(val_df.head())

print("\nSaving training data...")
train_df.write_parquet(rf".\Data\final_train_data2.parquet")
train_df.write_csv(rf".\Data\final_train_data2.csv")
print("Saving validation data...")
val_df.write_parquet(rf".\Data\final_val_data2.parquet")
val_df.write_csv(rf".\Data\final_val_data2.csv")