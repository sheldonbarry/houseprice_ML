import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# King County, Washington State, USA house data
# historic data of houses sold between May 2014 to May 2015
# https://github.com/dbendet/coursera_machine_learning/blob/master/kc_house_data.csv
data = pd.read_csv("kc_house_data.csv")

#print data.columns.values

# drop data with zero bedrooms and bathrooms and bedrooms outlier
data = data.drop(data[data.bedrooms == 0].index)
data = data.drop(data[data.bedrooms == 33].index)
data = data.drop(data[data.bathrooms == 0].index)
# drop columns that we won't be using
data = data.drop(['id', 'date', 'zipcode'], axis=1)

# create a dictionary of how featres correlate to house prices
correlations = {}
for column in data.loc[:, 'bedrooms':].columns:
	correlations[column] = data[column].corr(data['price'])

# setup grid for charts
fig, (ax1, ax2) = plt.subplots(1, 2)

# bar plot of correlations of features to price
ax1.bar(correlations.keys(), correlations.values())
ax1.title.set_text('Feature correlations')
ax1.set_ylabel('Correlation')
ax1.tick_params(labelrotation=90, labelsize=8)

# bar plot of most common number of bedrooms
x = data['bedrooms'].value_counts()
ax2.bar(x.index, x)
ax2.title.set_text('Number of bedrooms')
ax2.set_xlabel('Bedrooms')
ax2.set_ylabel('Count')
ax2.tick_params(labelsize=8)

# plot charts
fig.tight_layout()
plt.show()

# setup grid for charts
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)

# scatter plots of selected features vs price
ax1.scatter(data['bedrooms'], data['price'])
ax1.title.set_text('bedrooms')
ax2.scatter(data['bathrooms'], data['price'])
ax2.title.set_text('bathrooms')
ax3.scatter(data['sqft_living'], data['price'])
ax3.title.set_text('sqft_living')
ax4.scatter(data['grade'], data['price'])
ax4.title.set_text('grade')

# set label size for all subplots
for ax in fig.get_axes():
    ax.tick_params(labelsize=8)

# plot charts
plt.suptitle('Features vs price')
fig.tight_layout()
plt.show() 

# dependent variable
y = data['price']

# extract independent variables (all columns except price)
X = data.drop(['price'], axis=1)

# randomly split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state=10)

# initialise linear regression
reg = LinearRegression()

# fit training data
reg.fit(X_train, y_train)

# evaluate model
y_pred = reg.predict(X_test)
print ('Model evaluation:')
print ('R squared:\t {}'.format(r2_score(y_test, y_pred)))
print ('RMSE:\t\t {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
