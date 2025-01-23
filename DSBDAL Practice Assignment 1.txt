Name :- Omkar Ramesh Sawant
RollN0:- 13328

pip install numpy

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (1.26.4)
Note: you may need to restart the kernel to use updated packages.

import numpy as np
x=[2,3,4,5]
print(type(x))
y=np.array(x)
print(type(y))

<class 'list'>
<class 'numpy.ndarray'>


####Q1

runs = np.array([
    [45, 50, 30, 60, 25],    
    [10, 15, 5, 20, 12],     
    [80, 60, 75, 100, 90],   
    [25, 30, 20, 35, 40],    
    [0, 5, 10, 0, 2],        
    [50, 55, 60, 65, 70],    
    [35, 40, 45, 50, 55],    
    [100, 110, 120, 115, 125], 
    [10, 12, 15, 10, 8],     
    [70, 80, 85, 90, 95],    
    [40, 45, 50, 55, 60]     
])
total_runs = np.sum(runs, axis=1)
print("Total runs scored by each player:", total_runs)

highest_score = np.max(runs)
print("Highest score in a single match:", highest_score)

average_runs_per_player = np.mean(runs, axis=1)
print("Average runs scored per match by each player:", average_runs_per_player)


##OUTPUT

Total runs scored by each player: [210  62 405 150  17 300 225 570  55 420 250]

Highest score in a single match: 125

Average runs scored per match by each player: [ 42.   12.4  81.   30.    3.4  60.   45.  114.   11.   84.   50. ]






####Q2

sales_data = np.array([
    [150, 200, 250, 300, 400, 350, 500],  
    [120, 180, 210, 240, 310, 280, 400],  
    [100, 130, 190, 220, 300, 270, 350],  
    [80,  90,  150, 180, 240, 220, 300],  
    [50,  60,  80,  100, 130, 120, 180]   
])
shape = sales_data.shape
print("Shape :", shape)

num_dimensions = sales_data.ndim
print("Number of dimensions :", num_dimensions)

data_type = sales_data.dtype
print("Data type of elements :", data_type)

total_elements = sales_data.size
print("Total number of elements:", total_elements)

memory_used = sales_data.nbytes
print("Memory used :", memory_used)

product_3_day_4 = sales_data[2, 3]  
print("Sales data for Product 3 on Day 4:", product_3_day_4)

product_1_sales = sales_data[0, :]
print("All sales data for Product 1:", product_1_sales)

day_5_sales = sales_data[:, 4] 
print("All sales data for Day 5:", day_5_sales)

first_3_products_sales = sales_data[:3, :]
print("Sales data for first 3 products:\n", first_3_products_sales)

last_2_days_sales = sales_data[:, -2:]  
print("Sales data for last 2 days:\n", last_2_days_sales)

transposed_sales_data = sales_data.T 
print("Transposed sales data:\n", transposed_sales_data)



##OUTPUT

Shape : (5, 7)

Number of dimensions : 2

Data type of elements : int32

Total number of elements : 35

Memory used by the: 140

Sales data for Product 3 on Day 4: 220

All sales data for Product 1: [150 200 250 300 400 350 500]

All sales data for Day 5: [400 310 300 240 130]

Sales data for the first 3 products:
 [[150 200 250 300 400 350 500]
 [120 180 210 240 310 280 400]
 [100 130 190 220 300 270 350]]

Sales data for the last 2 days:
 [[350 500]
 [280 400]
 [270 350]
 [220 300]
 [120 180]]

Transposed sales data:
 [[150 120 100  80  50]
 [200 180 130  90  60]
 [250 210 190 150  80]
 [300 240 220 180 100]
 [400 310 300 240 130]
 [350 280 270 220 120]
 [500 400 350 300 180]]

