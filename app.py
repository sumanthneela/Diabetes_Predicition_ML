# prompt: generate taking input from the user and predict 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset (assuming it's in the same directory)
data_set = pd.read_csv('Diabetes_Dataset.csv')

# Separate features (x) and target (y)
x = data_set.iloc[:, :-1]
y = data_set.iloc[:, -1]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the SVM model
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)


# Function to get user input and make a prediction
def predict_diabetes():
    print("\nPlease Enter the following details: *** Note: Only enter numerical values ***\n")

    try:
        input_dy1 = float(input("Pregnancies (whole number): "))
        input_dy2 = float(input("Glucose: "))
        input_dy3 = float(input("Blood Pressure: "))
        input_dy4 = float(input("Skin Thickness: "))
        input_dy5 = float(input("Insulin: "))
        input_dy6 = float(input("BMI (e.g., 25.5): "))
        input_dy7 = float(input("Diabetes Pedigree Function (e.g., 0.627): "))
        input_dy8 = float(input("Age (whole number): "))

        input1 = np.array([[input_dy1, input_dy2, input_dy3, input_dy4,
                             input_dy5, input_dy6, input_dy7, input_dy8]])

        input_standardized = sc.transform(input1)
        Sol = model.predict(input_standardized)

        if Sol == 1:
            print("\nYou are diabetic. Please consult a doctor for proper medication and guidance.")
            print("⚡ Tips for managing diabetes:")
            print("   - Maintain a balanced diet with low sugar and carbs.")
            print("   - Exercise regularly to keep your blood sugar levels stable.")
            print("   - Monitor glucose levels and follow medical advice.")
            print("   - Manage stress and ensure proper sleep.")
        else:
            print("\nYou are not diabetic. Keep following a healthy lifestyle to stay safe!")

    except ValueError:
        print("Invalid input. Please enter numerical values only.")

# Call the function to start the prediction process
predict_diabetes()
