import os
import polars as pl # Using polars instead of pandas for speed. >9 million lines in 784k csv files.

# Calculate student_streak and student_ben_score


# student_streak is a measurement correct or incorrect answers in a row. 
# Positive values represent correct answers in a row while negative values 
# represent wrong answers in a row; for example, if a student correctly 
# answers 3 questions in a row, their student_streak gets incremented by 3. 
# Then, if they miss a question, their streak gets reset to 0 and then 
# decremented by one, leaving a student_streak of -1.

def calculate_streak(df):
    """Calculate student streak using Polars vectorized operations for memory efficiency"""
    print("Sorting data by student_id and timestamp...")
    df = df.sort(['student_id', 'timestamp'])
    
    print("Detecting student changes...")
    # Detect when student ID changes or correctness changes
    df = df.with_columns(
        pl.when(pl.col('student_id') != pl.col('student_id').shift(1))
        .then(1)
        .when(pl.col('correct') != pl.col('correct').shift(1))
        .then(1)
        .otherwise(0)
        .alias('streak_changed')
    )
    
    print("Creating groups for streaks...")
    # Create groups for each streak (same student, same correctness)
    df = df.with_columns(
        pl.col('streak_changed').cum_sum().alias('streak_group_id')
    )
    
    print("Computing streaks...")
    # Calculate streak within each group
    df = df.with_columns(
        pl.when(pl.col('correct') == 1)
        .then(pl.col('correct').cum_sum().over('streak_group_id'))
        .otherwise(-pl.arange(1, pl.len() + 1).over('streak_group_id'))
        .alias('student_streak')
    )
    
    print("Cleaning up temporary columns...")
    # Drop temporary columns
    df = df.drop(['streak_changed', 'streak_group_id'])
    
    return df


# UNCOMMENT STUFF AS NEEDED, MY COMPUTER DIDN'T HAVE ENOUGH RAM TO DO BOTH AT THE SAME TIME

# We split our data into 2 separate sets in the prep_data file
# print("Loading training data...")
# train_df = pl.read_parquet(rf".\Data\train_data.parquet")
print("Loading validation data...")
val_df = pl.read_parquet(rf".\Data\val_data.parquet")

# print("\nCalculating streak for training data...")
# train_df = calculate_streak(train_df)
print("\nCalculating streak for validation data...")
val_df = calculate_streak(val_df)

print("\nTraining data sample:")
print(val_df.head())

# print("\nSaving training data...")
# train_df.write_parquet(rf".\Data\final_train_data.parquet")
# train_df.write_csv(rf".\Data\final_train_data.csv")
print("Saving validation data...")
val_df.write_parquet(rf".\Data\final_val_data.parquet")
val_df.write_csv(rf".\Data\final_val_data.csv")