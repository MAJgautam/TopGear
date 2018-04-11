
# Exploratory Data Analysis using Python3  - Ajay Gautam M

## Dataset -  [Automobile Data set](https://archive.ics.uci.edu/ml/datasets/automobile) (UCI Machine learning Repository)

### Abstract: 
From 1985 Ward's Automotive Yearbook

### Dataset Information:
This data set consists of three types of entities: 
- the specification of an auto in terms of various characteristics 
- its assigned insurance risk rating
- its normalized losses in use as compared to other cars.

The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process **symboling.** A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. 

The third factor is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/speciality, etc...), and represents the average loss per car per year. 

### Attribute Information:

**Attribute: Attribute Range** 

1. symboling: -3, -2, -1, 0, 1, 2, 3. 
2. normalized-losses: continuous from 65 to 256. 
3. make: alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo 
4. fuel-type: diesel, gas. 
5. aspiration: std, turbo. 
6. num-of-doors: four, two. 
7. body-style: hardtop, wagon, sedan, hatchback, convertible. 
8. drive-wheels: 4wd, fwd, rwd. 
9. engine-location: front, rear. 
10. wheel-base: continuous from 86.6 120.9. 
11. length: continuous from 141.1 to 208.1. 
12. width: continuous from 60.3 to 72.3. 
13. height: continuous from 47.8 to 59.8. 
14. curb-weight: continuous from 1488 to 4066. 
15. engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor. 
16. num-of-cylinders: eight, five, four, six, three, twelve, two. 
17. engine-size: continuous from 61 to 326. 
18. fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi. 
19. bore: continuous from 2.54 to 3.94. 
20. stroke: continuous from 2.07 to 4.17. 
21. compression-ratio: continuous from 7 to 23. 
22. horsepower: continuous from 48 to 288. 
23. peak-rpm: continuous from 4150 to 6600. 
24. city-mpg: continuous from 13 to 49. 
25. highway-mpg: continuous from 16 to 54. 
26. price: continuous from 5118 to 45400.


## Preliminary Step


```python
#Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
sns.set(style="ticks")
```


```python
#Importing the Dataset
auto = pd.read_csv("...//input.csv")
```


```python
#Displaying the first 5 datapoints
auto.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.4</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.4</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



**Inference:** *From the inital look at the data, we can see that null values are represented as "?" in this dataset.*


```python
#Checking the dimensions of the dataset
auto.shape
```




    (205, 26)




```python
#Looking at the datatypes of the features
auto.dtypes
```




    symboling              int64
    normalized-losses     object
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                  object
    stroke                object
    compression-ratio    float64
    horsepower            object
    peak-rpm              object
    city-mpg               int64
    highway-mpg            int64
    price                 object
    dtype: object




```python
#Getting the descriptive statistics of numeric columns
auto.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>compression-ratio</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.834146</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>10.142537</td>
      <td>25.219512</td>
      <td>30.751220</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.245307</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>3.972040</td>
      <td>6.542142</td>
      <td>6.886443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>8.600000</td>
      <td>19.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>9.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>9.400000</td>
      <td>30.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>23.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Getting the descriptive statistics of numeric columns
auto.describe(include=[np.object])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>engine-type</th>
      <th>num-of-cylinders</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>52</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>8</td>
      <td>39</td>
      <td>37</td>
      <td>60</td>
      <td>24</td>
      <td>187</td>
    </tr>
    <tr>
      <th>top</th>
      <td>?</td>
      <td>toyota</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>ohc</td>
      <td>four</td>
      <td>mpfi</td>
      <td>3.62</td>
      <td>3.4</td>
      <td>68</td>
      <td>5500</td>
      <td>?</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>41</td>
      <td>32</td>
      <td>185</td>
      <td>168</td>
      <td>114</td>
      <td>96</td>
      <td>120</td>
      <td>202</td>
      <td>148</td>
      <td>159</td>
      <td>94</td>
      <td>23</td>
      <td>20</td>
      <td>19</td>
      <td>37</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning
From the preliminary look at the data, we know that there are missing values across diferent attributes in the dataset. So, we are gonna do imputation for those variables. 
We are replacing **?** with **NaN** because *Python3* recognises **NaN** as missing values by default. This step will enable to us to do an easy imputation.


```python
#Replacing "?" with "NaN" from the numpy library
auto = auto.replace("?", np.NaN)

#Checking the missing values count by attributes
auto.isnull().sum()
```




    symboling             0
    normalized-losses    41
    make                  0
    fuel-type             0
    aspiration            0
    num-of-doors          2
    body-style            0
    drive-wheels          0
    engine-location       0
    wheel-base            0
    length                0
    width                 0
    height                0
    curb-weight           0
    engine-type           0
    num-of-cylinders      0
    engine-size           0
    fuel-system           0
    bore                  4
    stroke                4
    compression-ratio     0
    horsepower            2
    peak-rpm              2
    city-mpg              0
    highway-mpg           0
    price                 4
    dtype: int64



### Mode Imputation
Since, the num-of-doors vairable has only 2 unique values. We will use the value with the **highest mode** for imputation.


```python
#Imputing the "num-of-doors" variable - mode imputation
auto['num-of-doors'] = auto['num-of-doors'].fillna("four")
```

### Mean Imputation
We will use **mean imputation** for the other vairables (numerical) with missing values. 


```python
#Imputing the variables having missing values - Mean Imputation
#Using "scikit-learn" library of Python

#Defining the imputation strategy and creating an instance
imp = Imputer(missing_values="NaN", strategy="mean")

#Running a 'for' loop to impute the variables containing missing values
for i in auto.columns:
    if auto[i].isnull().sum() > 0:
        auto[i] = imp.fit_transform(auto[[i]])

```


```python
#Checking - if the variables are imputed
auto.isnull().sum()
```




    symboling            0
    normalized-losses    0
    make                 0
    fuel-type            0
    aspiration           0
    num-of-doors         0
    body-style           0
    drive-wheels         0
    engine-location      0
    wheel-base           0
    length               0
    width                0
    height               0
    curb-weight          0
    engine-type          0
    num-of-cylinders     0
    engine-size          0
    fuel-system          0
    bore                 0
    stroke               0
    compression-ratio    0
    horsepower           0
    peak-rpm             0
    city-mpg             0
    highway-mpg          0
    price                0
    dtype: int64



## Univariate Analysis

### Vehicle Make  - Frequency Plot


```python
#Setting the plot parameters
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

#Vehicle Make - Frequency Plot
f, axis = plt.subplots(figsize=(15, 7))
sns.countplot(y='make', data=auto)
plt.title("Vehicle frequency by Make")
plt.ylabel('Make')
plt.xlabel('Count')
```




    Text(0.5,0,'Count')




![png](output_19_1.png)


**Inference:** *Toyota manufacturer has most number of vehicles with more than 40% than the 2nd highest maker Nissan. Mazda comes in as the 3rd highest make.*

### Fuel Type - Frequency Plot


```python
#Fuel type - frequency chart
f, axis = plt.subplots(figsize=(7, 7))
sns.countplot(x='fuel-type', data=auto)
plt.title("Fuel type frequency ")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type')

```




    Text(0.5,0,'Fuel type')




![png](output_22_1.png)


**Inference:** *'Gas' is preferred by more than 80% of the customers than 'diesel'.*

### No. Of Doors - Bar Chart


```python
#No. Of Doors - Bar Chart
f, axis = plt.subplots(figsize=(7, 7))
sns.countplot(x='num-of-doors', data=auto)
plt.title("No. of doors frequency plot")
plt.ylabel('Number of vehicles')
plt.xlabel('Number of doors');
```


![png](output_25_0.png)


**Inference:** *Even though there is not much difference, looks like 'four door vehicles' are popular among the customers.*

### Drive Wheels - Bar Chart


```python
#Drive Wheel - Bar Chart
f, axis = plt.subplots(figsize=(7, 7))
sns.countplot(x='drive-wheels', data=auto)
plt.title("Drive wheels frequency plot")
plt.ylabel('Number of vehicles')
plt.xlabel('Drive wheels');
```


![png](output_28_0.png)


**Inference:** *'Front Wheen Drive (fwd)' has the most number of cars followed by 'rear wheel drive (rwd)' and 'four wheel drive (4wd)'.*

### Insurance Risk Rating - Histogram


```python
#Insurance Risk Rating - Histogram 
auto.symboling.hist(bins=6)
plt.title("Insurance risk ratings of vehicles histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating');
```


![png](output_31_0.png)


**Inference:** *Insurance Risk rating follows a normal distubution with most number of vehicles in the range of ratings 0 and 1.*

### Normalized Losses - Histogram


```python
#Nomalized losses - Histogram
auto['normalized-losses'].hist(bins=6,color='tomato')
plt.title("Normalized losses of vehicles histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses');

```


![png](output_34_0.png)


**Inference:** *Normalized losses (the average loss payment per insured vehicle year) has more number of cars in the range of 65 and 150.*

### Curb-Weight - Density Plot


```python
#Curb Weight - Density Plot
sns.distplot(auto['curb-weight'])
plt.title("Curb weight density plot")
plt.ylabel('Number of vehicles')
plt.xlabel('Curb weight')
```




    Text(0.5,0,'Curb weight')




![png](output_37_1.png)


**Inference:** *Curb weight of the cars are distributed between 1500 and 4000 approx. The normal distibution in this case is slightly skewed to the right*

### Hosrpower - Histogram


```python
#Horse Power - Histogram
auto['horsepower'].hist(bins=6,color='mediumorchid')
plt.title("Horsepower histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Horsepower')
```




    Text(0.5,0,'Horsepower')




![png](output_40_1.png)


**Inference:** *With regarding to the horsepower, most vehicles are in between 50 and 125.*

## Correlation Analysis

### Correlation matrix - Heat Map


```python
#Correlation analysis - Heat Map
plt.figure(figsize=(15,13))
sns.heatmap(auto.corr(),annot=True, cmap = 'rocket_r')
plt.title('Correlation Matrix - Heat Map')
```




    Text(0.5,1,'Correlation Matrix - Heat Map')




![png](output_43_1.png)


**Inferences:**
- **price** is more correlated with **engine-size** and **curb-weight** among the other attributes. So, these two will be the **major predictors** in the model.
- **highway-mpg** is *highly correlated* with **city-mpg** with a correlation coefficient of 0.97. This can lead to severe **multicolliearity.** Since, these two variables are providing the same information for the model prediction, we can drop either one of these variables. 
- We can see that **curb-weight** have a *high correlation* with **engine-size**, **length**, **width** and **wheel-base** which is expected as these adds up to the weight of the car.
- Since, **wheel-base** of a car is determined by its **length** and **weight**, we can see the *high correlation* between these variables. 
- There is also a *high correlation* between **engine-size** and **horsepower**
- The target variable **price** has a stronger correlation with the following predictors,
    - engine-size
    - curb-weight
    - horsepower
    - city-mpg/highway-mpg
    - length/width


## Bivariate Analysis

### Price Vs Engine-size


```python
#Scatter plot - Price Vs Engine-size
sns.lmplot("engine-size",'price', auto)
plt.title("Price Vs Engine-size")
```




    Text(0.5,1,'Price Vs Engine-size')




![png](output_46_1.png)


**Inference:** *Higher the engine-size is, costlier the vehicle is.*

### Curb-weight Vs City-mpg and Highway-mpg


```python
#Scatter plot - Curb-weight Vs City-mpg
sns.lmplot('city-mpg',"curb-weight", auto)
plt.title("Curb-weight Vs City-mpg")

#Scatter plot - Curb-weight Vs Highway-mpg
sns.lmplot('highway-mpg',"curb-weight", auto)
plt.title("Curb-weight Vs Highway-mpg")
```




    Text(0.5,1,'Curb-weight Vs Highway-mpg')




![png](output_49_1.png)



![png](output_49_2.png)


**Inference:** *It is evident from the above plots that, 'city-mpg' and 'highway-mpg' have a negative correlation with the 'curb-weight'. It can be stated that, heavier vehicles give less mileage in both city and highway.*

### Price and Make - Box plot


```python
#Box-plot of Price and Make
plt.rcParams['figure.figsize']=(25,10)
ax = sns.boxplot(x="make", y="price", data=auto)
plt.title("Boxplot of Price and Make")
```




    Text(0.5,1,'Boxplot of Price and Make')




![png](output_52_1.png)


**Inferences:**
- The most expensive car is manufactured by **Mercedes-Benz**
- The lease expensive car is manufacture by **Chevrolet**
- BMW, Jaguar, Porsche and Mercedes-Benz are the manufaturers who make only **premium cars** in the high price range of above 20000.
- **least expensive affordable cars** in the price range of less than 10000 are manufatured by Chevrolet, Dodge, Honda, Mitsubishi, Plymoth and Subaru.

### Price and Drive-Wheels - Box plot


```python
#Box-plot of Price and Drive-Wheels
plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=auto)
plt.title("Box-plot of Price and Drive-wheels")
```




    Text(0.5,1,'Box-plot of Price and Drive-wheels')




![png](output_55_1.png)


**Inference:** *From the above plot, it can be seen that 'rear wheel drive (rwd)' cars are the most expensive and the 'front wheel drive (fwd)' cars are the least expensive. *

## Conclusion

The above **EDA** helped us to understand and obtain some key insights of the **Automobile dataset**. These insights will be extremely helpful in buidling a suitable model for prediction.

The following are the steps done in this **EDA**:
- **Prelimiary step** - looking at the **structure** and **shape** of the data
- **Data Cleaning** - finding and **imputing** the missing values
- **Univariate Analysis**  - finding the **patterns** and **summarizing** the data through different attributes using *frequency, histogram* and *density plots.*
- **Correlation Analysis** - finiding the **nature of relationship** between the different features and also the *strong,* *weak* and *reduntant predictors* associated with the target variable
- **Bivariate Analysis** - finding and **analysing the relationship** between various pairs of features through *box-plot and scatter plot.* 
