import numpy as np
import pandas as pd

# » reading data, defining its columns and index name
columns = 'height weight shoe_size gender'.split()
data = pd.DataFrame(
    data=np.loadtxt('data/gender_data.txt'),
    columns=columns)
data.index.name = 'id'

# » separating traning data and isolating gender
training_data = data.drop('gender', axis=1)
gender = data['gender']

# » columns that contain the value -1 at any row
columns_with_missing = [col for col in training_data.columns
                        if -1 in training_data[col].values]

# » replacing missing values with mean value of column
for col in columns_with_missing:
    col_values = training_data[col]
    invalid_indexes = np.where(col_values == -1)[0]
    valid_indexes = np.where(col_values != -1)[0]
    mean = np.mean(col_values[valid_indexes])
    col_values[invalid_indexes] = mean

# » dividing males and female indexes
id_female = np.where(gender == 0)[0]
id_male = np.where(gender == 1)[0]

# » calculating mean of each column values
# » in traning data, separate betwwen male 
# » and female
female_mean_height, female_mean_weight, female_mean_shoe_size = [
    training_data[col][id_female].mean()
    for col in training_data.columns]
male_mean_height, male_mean_weight, male_mean_shoe_size = [
    training_data[col][id_male].mean()
    for col in training_data.columns]

n_rows = data.shape[0]

# » mean of means
both_height_mean = (female_mean_height + male_mean_height) / 2
# » classifying as female(0) if value is
# » lower than `mean of means` and as 
# » male(1) if higher
height_classifier = 0+(training_data['height'] > both_height_mean)
# » finding score, by comparing the classifier
# » and the correct gender
height_score = 0+(height_classifier == gender).sum() / n_rows

both_weight_mean = (female_mean_weight + male_mean_weight) / 2
weight_classifier = 0+(training_data['weight'] > both_weight_mean)
weight_score = 0+(weight_classifier == gender).sum() / n_rows

both_shoe_size_mean = (female_mean_shoe_size + male_mean_shoe_size) / 2
shoe_size_classifier = 0+(training_data['shoe_size'] > both_shoe_size_mean)
shoe_size_score = 0+(shoe_size_classifier == gender).sum() / n_rows

# » finding the index of the most
# » accurate predictor
#                               0               1               2
best_predictor = np.argmax([height_score, weight_score, shoe_size_score])

print(f'height model: {height_score}      **missed 50% of predictions**')
print(f'weight model: {weight_score}     **missed 26% of predictions**')
print(f'shoe size model: {shoe_size_score}  **missed 16% of predictions** MOST ACCURATE MODEL!')