import os
import pandas as pd

# Define the paths and corresponding labels
class_paths = {
     r"E:\7th Semester\Crops\jute\pjute": 0,
     r"E:\7th Semester\Crops\paddy\ppaddy": 1,
     r"E:\7th Semester\Crops\sugarcane\psugercane": 2,
     r"E:\7th Semester\Crops\wheat\pwheat": 3,
     r"E:\7th Semester\Crops\vutta\pvutta": 4,
     r"E:\7th Semester\Crops\potato\ppotato": 5,
     r"E:\7th Semester\Crops\lentil\plentil": 6,
     r"E:\7th Semester\Crops\chilli\pchilli": 7,
     r"E:\7th Semester\Crops\mustard\pmustard": 8,
     r"E:\7th Semester\Crops\onion\ponion": 9


}

# Create an empty list to hold DataFrames
dfs = []

# Iterate through class paths and label the datasets
for path, label in class_paths.items():
    filenames = os.listdir(path)
    data = [{"path": os.path.join(path, filename), "label": label} for filename in filenames]
    class_df = pd.DataFrame(data)
    dfs.append(class_df)

# Concatenate the list of DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Shuffle the merged dataset
merged_df = merged_df.sample(frac=1).reset_index(drop=True)

# Save the merged dataset as a CSV file
merged_df.to_csv("cropsdataset.csv", index=False)
