# Todd Bartoszkiewicz
# CSC525: Introduction to Machine Learning
# Module 2: Critical Thinking Option #2
#
# Option #2: Predicting Video Game Preferences with KNN
# KNN cluster classification works by finding the distances between a query and all examples in its data. The specified
# number of examples (K) closest to the query are selected. The classifier then votes for the most frequent label found.
#
# There are several advantages of KNN classification, one of them being simple implementation. The search space is
# robust as classes do not need to be linearly separable. It can also be updated online easily as new instances with
# known classes are presented.
#
# A KNN model can be implemented using the following steps:
#
# Load the data;
# Initialize the value of k;
# For getting the predicted class, iterate from 1 to the total number of training data points;
# Calculate the distance between the test data and each row of training data;
# Sort the calculated distances in ascending order based on distance values;
# Get top k rows from the sorted array;
# Get the most frequent class of these rows; and
# Return the predicted class.
#
# For your assignment, you will build a KNN classifier in Python.
#
# Download the class data in CSV format:
# https://gist.githubusercontent.com/dhar174/14177e1d874a33bfec565a07875b875a/raw/7aa9afaaacc71aa0e8bc60b38111c24e584c74d8/data.csv
#
# Using this data, your classifier should be able to predict a person's favorite video game genre based on their age,
# height, weight, and gender. (Do not worry about real-world accuracy here. This is to provide you with an opportunity
# to practice.)
#
# You can also choose to collect and use your own data points.
#
# Submission should include an executable Python file that accepts input of 4 floating point numbers representing,
# respectively, age (in years), height (in inches), weight (in lbs), and gender (females represented by 0s and males
# represented by 1s).
import csv
import urllib.request
import math
import os
from collections import Counter


DATA_URL = "https://gist.githubusercontent.com/dhar174/14177e1d874a33bfec565a07875b875a/raw/7aa9afaaacc71aa0e8bc60b38111c24e584c74d8/data.csv"
DATA_FILE = "data.csv"


# I'm guessing that you don't want this data submitted with everyone's submission, so including a little helper
# method to download the data from the site if needed
def download_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading data...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
            print(f"Data saved as {DATA_FILE} successfully.")
        except Exception as ex:
            print(f"Error downloading data file: {ex}")
            exit(1)


if __name__ == "__main__":
    print("Welcome to Predicting Video Game Preferences with KNN")
    input_string = input("Please enter 4 floating point numbers representing age (in years), height (in inches), weight (in lbs), and gender (females represented by 0s and males represented by 1s), separated by spaces\nFor example:\n29.5 60.5 150.5 1\n")
    values = list(map(float, input_string.split()))

    if len(values) != 4:
        print("We're missing one of the four input numbers. Please run the script again.")
    else:
        # A KNN model can be implemented using the following steps:
        # 1. Load the data;
        download_data()
        data = []
        labels = []
        try:
            with open(DATA_FILE, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    # age, height, weight, gender
                    data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                    # genre
                    labels.append(row[4])
        except Exception as exc:
            print(f"Error reading data: {exc}")
            exit(1)

        # 2. Initialize the value of k;
        k = 3

        distances = []

        # 3. For getting the predicted class, iterate from 1 to the total number of training data points;
        for i, data_point in enumerate(data):
            # 4. Calculate the distance between the test data and each row of training data;
            distance = math.dist(values, data_point)
            distances.append((distance, labels[i]))

        # 5. Sort the calculated distances in ascending order based on distance values;
        distances.sort(key=lambda x: x[0])

        # 6. Get top k rows from the sorted array;
        top_k_rows = [label for _, label in distances[:k]]

        # 7. Get the most frequent class of these rows; and
        most_common = Counter(top_k_rows).most_common(1)

        # 8. Return the predicted class.
        predicted_genre = most_common[0][0]

        print(f"Predicted favorite video game genre: {predicted_genre}")
