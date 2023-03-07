import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
# TODO: Create model to classifiy EEG motor actions

# Printing out data just to see if the import worked
# with open('Data/ Prithvi-LE.txt', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         print(row)
#         line_count += 1
#     print(f'Processed {line_count} lines.') 

# Load EEG signal and base pattern
eeg_signal = np.load('Data/ Prithvi-LE.txt')
base_pattern = np.load('Data/OpenBCI-RAW-Baseline.txt')

# Define time window for base pattern
# This is the toughest part to algorithmically set. Might have to use multiple batches of filtering 
# but this is a template so far 
window_start = 1000
window_end = 1500
base_pattern = base_pattern[window_start:window_end]

# Subtract base pattern from EEG signal
filtered_signal = eeg_signal - np.tile(base_pattern, int(len(eeg_signal)/len(base_pattern))+1)[:len(eeg_signal)]

# Plot original and filtered signals
plt.plot(eeg_signal, label='Original')
plt.plot(filtered_signal, label='Filtered')
plt.legend()
plt.show()


# Template code for the Random Forest Classifier algorithm using the dataset
dataLE = pd.read_csv('Data/ Prithvi-LE.txt')

# Split the data into training and testing sets
train_data = dataLE.sample(frac=0.8, random_state=42)
test_data = dataLE.drop(train_data.index)

# Define the input and output variables for the algorithm
X_train = train_data.drop('target_variable_name', axis=1).values
y_train = train_data['target_variable_name'].values
X_test = test_data.drop('target_variable_name', axis=1).values
y_test = test_data['target_variable_name'].values

# Create a random forest classifier
rfclassifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Fit the model to the training data
rfclassifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfclassifier.predict(X_test)

# Evaluate the performance of the algorithm
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion matrix:')
print(conf_matrix)


