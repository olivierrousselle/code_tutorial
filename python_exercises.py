
""" Correction exercises """

""" Packages """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_wine
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

wine_data = load_wine()
df_wine = pd.DataFrame(data = np.c_[wine_data['data'], wine_data['target']], columns= wine_data['feature_names'] + ['target'])

X_train, X_test, y_train, y_test = train_test_split(df_wine.iloc[:, :-1], df_wine['target'], test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {round(accuracy,2)}")

""" Exercise 11 """

def calculer_moyenne_mobile(prix, fenetre):
    moyenne_mobile = []
    for i in range(len(prix) - fenetre + 1):
        moyenne = np.mean(prix[i:i+fenetre])
        moyenne_mobile.append(moyenne)
    return moyenne_mobile

def trouver_points_achat_vente(prix, moyenne_mobile):
    points_achat = []
    points_vente = []
    for i in range(1, len(prix)):
        if prix[i] > moyenne_mobile[i-1] and prix[i-1] <= moyenne_mobile[i-1]:
            points_achat.append(i)
        elif prix[i] < moyenne_mobile[i-1] and prix[i-1] >= moyenne_mobile[i-1]:
            points_vente.append(i)
    return points_achat, points_vente

prix = [100, 105, 110, 115, 120, 115, 110, 105, 100, 95]
fenetre_moyenne_mobile = 3

moyenne_mobile = calculer_moyenne_mobile(prix, fenetre_moyenne_mobile)
points_achat, points_vente = trouver_points_achat_vente(prix, moyenne_mobile)

jours = range(len(prix))
plt.plot(jours, prix, label='Prix', color='blue')
plt.plot(jours[fenetre_moyenne_mobile-1:], moyenne_mobile, label=f'Moyenne mobile ({fenetre_moyenne_mobile} jours)', color='red', linestyle='--')
plt.scatter(points_achat, [prix[i] for i in points_achat], color='green', label='Point d\'achat', marker='^', s=100)
plt.scatter(points_vente, [prix[i] for i in points_vente], color='red', label='Point de vente', marker='v', s=100)
plt.title('Stratégie de suivi de tendance')
plt.xlabel('Jours')
plt.ylabel('Prix')
plt.legend()
plt.grid(True)
plt.show()
