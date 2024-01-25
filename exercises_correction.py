
""" Correction exercises """

""" Packages """

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pandas as pd
import ccxt
import ta
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


""" Exercise 1 """

name = "YourName"
age = 25
print(f"My name is {name} and I am {age} years old.")

grades_list = [15, 18, 12]
average = sum(grades_list) / len(grades_list)
print(f"The average of the grades is {average}.")

user_age = int(input("Enter your age: "))
if user_age > age:
    print("You are older than me!")
elif user_age < age:
    print("You are younger than me!")
else:
    print("We are the same age!")
    

""" Exercise 2 """

groceries = ["bread", "milk", "eggs"]
for index in range(len(groceries)):
    print(f"{index+1}. {groceries[index]}")
    
groceries.append("cheese")
index = 0
while index < len(groceries):
    print(groceries[index])
    index += 1
    
budget = 20
total = sum([5, 3, 2, 4])
if total <= budget:
    print("You can afford all items.")
else:
    print("You cannot afford all items with your budget.")


""" Exercise 3 """

import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()

matrix[matrix % 2 == 0] *= 2
print(matrix)

column_sums = matrix.sum(axis=0)
print("Sum of each column:", column_sums)    
    

""" Exercise 4 """

import pandas as pd

data = {"Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 22],
        "City": ["Paris", "New York", "London"]}
df = pd.DataFrame(data)
print(df)

filtered_df = df[df["Age"] > 25]
print(filtered_df)

df["Name_Length"] = df["Name"].apply(len)
print(df)


""" Exercise 5 """

from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

print(df_iris.head())

class_means = df_iris.groupby('target').mean()
print("Class Means:\n", class_means)


""" Exercise 6 """

user_number = int(input("Enter a number: "))
if user_number > 0:
    print("The number is positive.")
elif user_number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")
    
if user_number % 2 == 0 and user_number > 0:
    print("The number is positive and even.")
    

""" Exercise 7 """

power_of_two = 2
while power_of_two <= 64:
    print(power_of_two, end=" ")
    power_of_two *= 2

names = ["Alice", "Bob", "Charlie"]
for name in names:
    print(len(name))

squares = [num ** 2 for num in range(1, 11)]
print(squares)


""" Exercise 8 """

def sum_of_squares(n):
    return sum(i ** 2 for i in range(1, n+1))

result = sum_of_squares(5)
print("Sum of squares:", result)

def sum_of_powers(n, power):
    return sum(i ** power for i in range(1, n+1))
result = sum_of_powers(5, 3)
print("Sum of cubes:", result)

""" Exercise 9 """

import matplotlib.pyplot as plt

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
sales = [1500, 2000, 1800, 2500, 3000, 2800, 2000, 2200, 2400, 2800, 3200, 3500]
plt.bar(months, sales)
plt.xlabel('Months')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.show()

categories = ["Category A", "Category B", "Category C"]
sales_distribution = [30, 45, 25]
plt.pie(sales_distribution, labels=categories, autopct='%1.1f%%')
plt.title('Sales Distribution by Category')
plt.show()

x_values = np.random.rand(10)
y_values = np.random.rand(10)
plt.scatter(x_values, y_values)
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


""" Exercise 10 """

data_box = [np.random.normal(0, std, 100) for std in range(1, 3)]
plt.boxplot(data_box)
plt.title('Box Plot for Random Data')
plt.show()

data_hist = np.random.random(1000)
plt.hist(data_hist, bins=30, edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

time_points = np.arange(0, 10, 0.1)
values = np.sin(time_points)
plt.scatter(time_points, values, marker='o', color='green')
plt.title('Sin Wave')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()


""" Exercise 11 """

from sklearn.datasets import load_wine

wine_data = load_wine()
df_wine = pd.DataFrame(data = np.c_[wine_data['data'], wine_data['target']], columns= wine_data['feature_names'] + ['target'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_wine.iloc[:, :-1], df_wine['target'], test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {round(accuracy,2)}")


