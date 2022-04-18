# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
  
# A custom function to calculate
# probability distribution function
def pdf(data):
    mean = np.mean(data)
    std = np.std(data)
    x = np.arange(-4 * std + mean, 4 * std + mean, 0.1)
    sorted(data)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (data - mean)**2 / (2 * std**2))
    return y_out, x
    
# To generate an array of x-values
# x = np.arange(-2, 2, 0.1)
data = [1.0279, 1.0147, 0.9820, 0.9924, 1.0674]
  
# To generate an array of
# y-values using corresponding x-values
y, x = pdf(data)
  
# Plotting the bell-shaped curve
plt.style.use('seaborn')
plt.figure(figsize = (6, 6))
plt.plot(x, y, color = 'black',
         linestyle = 'dashed')
  
plt.scatter( x, y, marker = 'o', s = 25, color = 'red')
plt.show()