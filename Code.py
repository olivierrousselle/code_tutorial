
""" Tutorial Python - Olivier Rousselle / UAP """

""" Modules """

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pandas as pd
from decimal import Decimal
import ccxt
import ta


""" Variables """

var = 10
print(var) # => 10

int_var = -3
float_var = 4.5
str_var = "Bitcoin"
bool_var = True
print(str_var) # => Bitcoin

result = -3 * 4.5
print(result) # => -13.5

word1 = "Hello"
word2 = " world"
print(word1 + word2) # => Hello world


dt = datetime.datetime(2024, 1, 20, 20, 30)
print(dt)  # => 2024-01-20 20:30:00

now = datetime.datetime.now()
print(now)  # => show the precise current moment

date_string = "2024-01-20"
date_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
print(date_object) # => 2024-01-20 20:30:00

timestamp = time.time()
print(timestamp) # show current timestamp


""" Structures of data """


my_list = ["BTC", 40000, "ETH", 2400]


print(my_list[0]) # => BTC
print(my_list[2]) # => ETH
print(my_list[-1]) # => 2400

my_list[2] = "ATOM"
print(my_list) # => ["BTC', 40000, 'ATOM', 2400]

len(my_list) # => 4

my_list.append("AVAX")
print(my_list) # => ["BTC', 40000, 'ATOM', 2400, 'AVAX']

my_list.insert(1, "ETF") # => ["BTC', 'ETF', 40000, 'ATOM', 2400, 'AVAX']

del my_list[1] # => ["BTC', 40000, 'ATOM', 2400, 'AVAX']

my_list_multidimensional = [["BTC", 40000, "ETH", 2400], ["AVAX", "ATOM"]]

print(my_list_multidimensional[0][1]) # => 40000


my_array = np.array([40000, 2400, 1])
my_array_2 = np.arange(0, 10, 3) # => array([0, 3, 6, 9])
my_array_3 = np.linspace(0, 10, 3) # => array([0, 5, 10])

print(my_array[0]) # => 40000
print(my_array_2.mean()) # => 4.5

my_array_multidimensional = np.array([[1, 2, 3],
                                      [5, 6, 7]])


assets_price = {"BTC": 40000, "ETH": 2400}

print(assets_price["BTC"]) # => 40000

assets_price["ETH"] = 2500
print(assets_price["ETH"]) # => 2500

assets_price["USDT"] = 1
print(assets_price) # => {'BTC': 40000, 'ETH': 2400, 'USDT': 1}

del assets_price["USDT"]


""" Dataframes """

my_array = np.array([["2024-01-01", 40000, 1000], 
                     ["2024-01-02", 41000, 1200],
                     ["2024-01-03", 40500, 1300],
                     ["2024-01-04", 41200, 1250],
                     ["2024-01-05", 40800, 1100]])
df = pd.DataFrame(my_array, columns = ['date', 'price', 'volume']) 

data = {"date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "price": [40000, 41000, 40500, 41200, 40800],
        "volume": [1000, 1200, 1300, 1250, 1100]}
df = pd.DataFrame(data)

print(df['price'])
print(df.iloc[0])

df.head()
df.shape # => (5,3)

df['date'] = pd.to_datetime(df['date'], unit='ms')
df = df.set_index(df['date'])
del df['date']

df.loc['2024-01-01'] # => row of date '2024-01-01'

df['new column'] = df['price'] * df['volume'] # new column price * volume

df['Variation price %'] = round(df['price'].pct_change()*100, 2)


print(df['price'].mean()) # => average of the prices
print(df['volume'].std()) # => standard deviation of the volumes


df_list = {"BTC": df, "ETH": df}


for index, row in df.iterrows():
    print(row)

""" Conditions and loops """

bitcoin_price = 41000 

if bitcoin_price <= 35000: 
    print("Buy")
elif bitcoin_price > 35000 and bitcoin_price <= 100000: 
    print("Hold") 
else: 
    print("Sell")
# => Hold   
    
list_coins = ["BTC", "ETH", "USDT", "AVAX", "ATOM"]
for coin in list_coins:
    print(coin)
# => BTC, ETH, USDT, AVAX, ATOM

for x in range(5):
    print(x) 
# => 0, 1, 2, 3, 4

balance = 10000
for i in range(12):
	balance = balance + 4000 - 1500
print("Balance at the end of the year :", balance, "$") # => 40000 $


initial_investment = 1000
years = 0
current_value = initial_investment
while current_value < initial_investment * 2:
    current_value = current_value * 1.03
    years += 1
print("Number of years:", years) # => Number of years: 24


""" Functions """

def calculate_area_rectangle(a, b):
  result = a * b
  return result

def calculate_area_rectangle(a : int, b : int) -> float:
  result = a * b
  return result

area = calculate_area_rectangle(2, 3)
print(area) # => 6



""" Graphs """

x = np.linspace(-2*np.pi, 2*np.pi, 50) # 50 numbers from -2π to 2π
y = np.sin(x) # function sinus
plt.plot(x, y) # plot of the curve
plt.scatter(x, y, color="red") # plot of the points (in red)
plt.xlabel("Axe X") # legend of x-axis
plt.ylabel("Axe Y") # legend of y-axis
plt.title("Graphique") # title of the graph
plt.show() # drawing of the graph


df['price'].plot()
plt.ylabel('price')
plt.show()


""" Trading """

api_key = ""
api_secret = ""
client = ccxt.binance({"apiKey": api_key, "secret": api_secret, "options": {'defaultType': 'spot'}})

klinesT = client.fetch_ohlcv('BTC/USDT:USDT', '15m', limit=100)

df = pd.DataFrame(np.array(klinesT)[:,:6], columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df = df.set_index(df['timestamp'])
df.index = pd.to_datetime(df.index, unit='ms')
del df['timestamp']


#client.createOrder('BTC/USDT:USDT', 'market', 'buy', amount, params={'leverage': 1})

df['MA10'] = ta.trend.sma_indicator(close=df['close'], window=10)

df['close'].plot(label='close')
df['MA10'].plot(label='MA10')
plt.legend()
plt.show()

""" Tests / Errors / exceptions """

number = 0
try:
    result = 5 / number
except:
    print("Error: division by 0 not possible")

a = 0
assert a==0
assert a==1 # => AssertionError    

        
        


