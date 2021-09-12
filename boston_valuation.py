from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIME_IDX, ZN_IDX, CHAS_IDX, RM_IDX, PTRATIO_IDX = 0, 1, 2, 4, 8
property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

def get_log_estimate(nr_rooms,
                     students_per_classroom,
                     next_to_river=False,
                     high_confidence=True):
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom    
    property_stats[0][CHAS_IDX] = 1 if next_to_river else 0
    
    # Make Prediction
    log_estimate = regr.predict(property_stats)[0][0]
    standard_deviation = RMSE * (2 if high_confidence else 1)
    
    # Calc Range
    upper_bound = log_estimate + standard_deviation
    lower_bound = log_estimate - standard_deviation
    interval  = 95 if high_confidence else 68
    
    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """Estimate price of a property in Boston.
    
    Parameters:
    -----------
    rm: number of rooms in the property
    
    ptratio: number of students per teacher in the classroom for the school in the area
    
    chas: True if the property is next to the river, False otherwise
    
    large_range: True for a 95% prediction interval, False for a 68% interval.
    

    """
    
    if rm < 1 or ptratio < 1:
        print('This is unrealistic. Try Again')
        return
    
    log_est, upper, lower, conf = get_log_estimate(nr_rooms = rm,
                                                   students_per_classroom = ptratio,
                                                   next_to_river = chas,
                                                   high_confidence = large_range)

    # Convert to today's dollars
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR

    # Round the dollar values to nearest thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi}, at the high end')