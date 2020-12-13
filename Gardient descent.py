#!/usr/bin/env python
# coding: utf-8

# ### Notebook import and Package
# 

# In[51]:


import matplotlib.pyplot as plt 
import numpy as np 

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm # color map

from sympy import symbols, diff
from math import log

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# # Example 
# f(x) = x^2 + x + 1

# In[2]:


def f(x):
    return x**2 + x + 1


# In[3]:


#Make data 
x_1 = np.linspace (start= -3, stop= 3, num = 100 )
x_1


# In[4]:


plt.xlim([-3, 3])
plt.ylim([0,8])
plt.xlabel('X', fontsize= 16)
plt.ylabel('f(x)', fontsize= 16)
plt.plot(x_1, f(x_1))
plt.show()


# ### slope & Derivatives 

# In[5]:


def df(x):
       return 2*x + 1 


# In[6]:


# plot function and derivate side by side

plt.figure(figsize=[10, 5])
# 1 Chart: Cost function
plt.subplot(1,2,1)

plt.xlim([-3, 3])
plt.ylim([0,8])

plt.title('Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('f(x)', fontsize= 16)

plt.plot(x_1, f(x_1), color='blue', linewidth = 5 )

# 2 Chart: Derivative 
plt.subplot(1,2,2)
plt.grid()
plt.title('Slope of the Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('df(x)', fontsize= 16)

plt.xlim([-2, 3])
plt.ylim([-3, 6])

plt.plot(x_1, df(x_1), color='skyblue', linewidth = 5)

plt.show()


# ### Python Loops & Gradient Descent 

# In[7]:


# python for loop 
for n in range(1):
    print('hello', n)
print('end of the loop')


# In[8]:


# while loop 

counter = 0

while counter < 1:
    print('counting....')
    counter = counter + 1
    


# In[9]:


# python for loop 
for n in range(1):
    print('hello', n)
print('end of the loop')


# In[10]:


# Gradient Descent 
new_x = 3 
previous_x = 0
step_mutiplier = 0.1
percision = 0.00000000000001 

x_list = [new_x]
slope_list = [df(new_x)]

for n in range(500):
    previous_x = new_x 
    gradient = df(previous_x)
    new_x = previous_x - step_mutiplier * gradient
    
    step_size = abs(new_x - previous_x)
    # print (step_size) 
    
    x_list.append(new_x)
    slope_list.append(df(new_x))
    
    
    if step_size < percision:
        print('loop ran this many times:', n)
        break
print('local minimum occurs at:', new_x)
print('slope oo df(x) value at this point is:', df(new_x))
print('f(x) value or cost at this point is ', f(new_x))


# In[11]:


# Superimpose the gradient desecent  calculation plot function and derivate side by side

plt.figure(figsize=[20, 7])
# 1 Chart: Cost function
plt.subplot(1,3,1)

plt.xlim([-3, 3])
plt.ylim([0,8])

plt.title('Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('f(x)', fontsize= 16)

plt.plot(x_1, f(x_1), color='blue', linewidth = 5 , alpha = 0.8)


value = np.array(x_list)

plt.scatter(x_list, f(value), color= 'red', s= 100, alpha = 0.6)

# 2 Chart: Derivative 
plt.subplot(1,3,2)
plt.grid()
plt.title('Slope of the Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('df(x)', fontsize= 16)

plt.xlim([-2, 3])
plt.ylim([-3, 6])

plt.plot(x_1, df(x_1), color='skyblue', linewidth = 5, alpha = 0.8)
plt.scatter(x_list, slope_list, color= 'red', s= 100, alpha = 0.6)

# 3 Chart: Derivative (close up)
plt.subplot(1,3,3)
plt.grid()
plt.title('Gradient Descent (close up)', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('df(x)', fontsize= 16)

plt.xlim([-0.55, -0.2])
plt.ylim([-0.3, 0.8])

plt.plot(x_1, df(x_1), color='skyblue', linewidth = 6, alpha = 0.8)
plt.scatter(x_list, slope_list, color= 'red', s= 300, alpha = 0.6)
plt.show()


# ### Multiple Minima Vs Intial Quess & Adanced Funtions
# # $$ g(x)= x^4 - 4x^2 + 5 $$

# In[12]:


# Make some data 
x_2 = np.linspace (start= -2, stop= 2, num = 100 )


# In[13]:


def g(x):
    return x**4 - 4*x**2 + 5


# In[14]:


def dg(x):
    return 4*x**3 - 8*x


# In[15]:


# plot function and derivate side by side

plt.figure(figsize=[20, 5])
# 1 Chart: Cost function
plt.subplot(1,2,1)

plt.xlim([-2, 2])
plt.ylim([0.5, 5.5])

plt.title('Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('g(x)', fontsize= 16)

plt.plot(x_2, g(x_2), color='blue', linewidth = 5 )

# 2 Chart: Derivative 
plt.subplot(1,2,2)
plt.grid()
plt.title('Slope of the Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('dg(x)', fontsize= 16)

plt.xlim([-2, 2])
plt.ylim([-6, 8])

plt.plot(x_2, dg(x_2), color='skyblue', linewidth = 5)

plt.show()


# In[16]:


# Gradient Descent 
def gradient_decent(derivative_func, intial_guess, multiplier = 0.02, percision = 0.0001, max_inter = 300):



    new_x = intial_guess 
    
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(max_inter):
        previous_x = new_x 
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient

        step_size = abs(new_x - previous_x)
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))


        if step_size < percision:
           
            break
    return new_x, x_list, slope_list


# In[17]:


local_min, list_x, deriv_list =  gradient_decent(derivative_func =dg, intial_guess = 0)

print('Local min occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[18]:


#calling gradient descent function
local_min, list_x, deriv_list =  gradient_decent(derivative_func =dg, intial_guess = 1)

# plot function and derivate and scatter plot side by side

plt.figure(figsize=[20, 5])
# 1 Chart: Cost function
plt.subplot(1,2,1)

plt.xlim([-2, 2])
plt.ylim([0.5, 5.5])

plt.title('Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('g(x)', fontsize= 16)

plt.plot(x_2, g(x_2), color='blue', linewidth = 5, alpha = 0.6 )
plt.scatter(list_x, g(np.array(list_x)), color = 'red', s=100, alpha= 0.6)

# 2 Chart: Derivative 
plt.subplot(1,2,2)
plt.grid()
plt.title('Slope of the Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('dg(x)', fontsize= 16)

plt.xlim([-2, 2])
plt.ylim([-6, 8])

plt.plot(x_2, dg(x_2), color='skyblue', linewidth = 5, alpha= 0.6)
plt.scatter( list_x, deriv_list, color = 'red', s=100, alpha= 0.6)


plt.show()


# ### Example 3 - Divergence, overflow and python Tuples
# 
# 
# $$ h(x) = x**5 - 2x**4 + 2 $$

# In[19]:


# Make data 

x_3 = np.linspace(start = -2.5, stop = 2.5, num= 1000)

def h(x):
    return x**5 - 2*x**4 + 2


def dh(x):
    return 5*x**4 - 8*x**3


# In[20]:


#calling gradient descent function
local_min, list_x, deriv_list =  gradient_decent(derivative_func =dh, intial_guess = - 0.2, max_inter =70)

# plot function and derivate and scatter plot side by side

plt.figure(figsize=[20, 5])
# 1 Chart: Cost function
plt.subplot(1,2,1)

plt.xlim([-1.2, 2.5])
plt.ylim([-1, 4 ])

plt.title('Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('h(x)', fontsize= 16)

plt.plot(x_3, h(x_3), color='blue', linewidth = 5, alpha = 0.6 )
plt.scatter(list_x, h(np.array(list_x)), color = 'red', s=100, alpha= 0.6)

# 2 Chart: Derivative 
plt.subplot(1,2,2)
plt.grid()
plt.title('Slope of the Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('dh(x)', fontsize= 16)

plt.xlim([-1, 2])
plt.ylim([-4, 5])

plt.plot(x_3, dh(x_3), color='skyblue', linewidth = 5, alpha= 0.6)
plt.scatter( list_x, deriv_list, color = 'red', s=100, alpha= 0.6)


plt.show()

print('local min occurs at:', local_min)
print('cost at this minimul is:',h(local_min))
print('number of steps:', len(list_x))


# In[21]:


# Calling gradient descent function
local_min, list_x, deriv_list = gradient_decent(derivative_func=dh, intial_guess= -0.2, 
                                                max_inter=71)

# Plot function and derivative and scatter plot side by side

plt.figure(figsize=[15, 5])

# 1 Chart: Cost function
plt.subplot(1, 2, 1)

plt.xlim(-1.2, 2.5)
plt.ylim(-1, 4)

plt.title('Cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('h(x)', fontsize=16)

plt.plot(x_3, h(x_3), color='blue', linewidth=3, alpha=0.8)
plt.scatter(list_x, h(np.array(list_x)), color='red', s=100, alpha=0.6)

# 2 Chart: Derivative
plt.subplot(1, 2, 2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dh(x)', fontsize=16)
plt.grid()
plt.xlim(-1, 2)
plt.ylim(-4, 5)

plt.plot(x_3, dh(x_3), color='skyblue', linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=0.5)

plt.show()

print('Local min occurs at: ', local_min)
print('Cost at this minimum is: ', h(local_min))
print('Number of steps: ', len(list_x))


# In[22]:


import sys

sys.float_info.max


# ###  The Learning rate
# 

# In[54]:


#calling gradient descent function
local_min, list_x, deriv_list =  gradient_decent(derivative_func =dg, intial_guess = 1.9, multiplier = 0.001 , max_inter=500)

# plot function and derivate and scatter plot side by side

plt.figure(figsize=[20, 5])
# 1 Chart: Cost function
plt.subplot(1,2,1)

plt.xlim([-2, 2])
plt.ylim([0.5, 5.5])

plt.title('Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('g(x)', fontsize= 16)

plt.plot(x_2, g(x_2), color='blue', linewidth = 5, alpha = 0.6 )
plt.scatter(list_x, g(np.array(list_x)), color = 'red', s=100, alpha= 0.6)

# 2 Chart: Derivative 
plt.subplot(1,2,2)
plt.grid()
plt.title('Slope of the Cost Function', Fontsize= 17)
plt.xlabel('X', fontsize= 16)
plt.ylabel('dg(x)', fontsize= 16)

plt.xlim([-2, 2])
plt.ylim([-6, 8])

plt.plot(x_2, dg(x_2), color='skyblue', linewidth = 5, alpha= 0.6)
plt.scatter( list_x, deriv_list, color = 'red', s=100, alpha= 0.6)


plt.show()

print('Number of step is:', len(list_x))


# In[55]:


#calling gradient descent function
n = 100

low_gamma =  gradient_decent(derivative_func =dg, intial_guess = 3, multiplier = 0.0005 ,percision = 0.0001, max_inter=n)

mid_gamma = gradient_decent(derivative_func =dg, intial_guess = 3, multiplier = 0.001 ,percision = 0.0001, max_inter=n)

high_gamma = gradient_decent(derivative_func =dg, intial_guess = 3, multiplier = 0.002 ,percision = 0.0001, max_inter=n)




plt.figure(figsize=[20, 10])
# 1 Chart: Cost function


plt.xlim([0, n])
plt.ylim([0, 50])

plt.title('Effect of the learning rate', Fontsize= 17)
plt.xlabel('Nr of iteration', fontsize= 16)
plt.ylabel('Cost', fontsize= 16)

#value for our chart low learing rate  
# 1) Y Axis Data: conert the list to numpy array 
low_values = np.array(low_gamma[1])

# 2) X Axis Data: create a list from 0 to n+1
iteration_list = list(range(0 , n+1))

#ploting a low learning rate
plt.plot(iteration_list, g(low_values), color='lightgreen', linewidth = 5, alpha = 0.6 )
plt.scatter(iteration_list, g(low_values), color = 'red', s=100, alpha= 0.6)



#ploting a mid learning rate
plt.plot(iteration_list, g(np.array(mid_gamma[1])), color='darkgreen', linewidth = 5, alpha = 0.6 )
plt.scatter(iteration_list, g(np.array(mid_gamma[1])), color = 'blue', s=100, alpha= 0.6)



#ploting a high learning rate
plt.plot(iteration_list, g(np.array(high_gamma[1])), color='hotpink', linewidth = 5, alpha = 0.6 )
plt.scatter(iteration_list, g(np.array(high_gamma[1])), color = 'pink', s=100, alpha= 0.6)

plt.show()


# # 4 Data Viz with 3D Charts
# # Minimise $$ f(x,y) = \frac{1} {3^{-x^2 - y^2}+ 1 }$$
# 

# In[24]:


def f(x,y):
    r = 3**(-x**2 - y**2)
    return 1/(r + 1)

def fg(x,y):
    v = 3**(-2**x - 2**y)
    return 1/(v+1)


# In[25]:


# Make our x and y data 

x_4 = np.linspace(start = -2, stop = 2, num=200 )
y_4 = np.linspace(start = -2, stop = 2, num=200 )

print('Shape of X array', x_4.shape)


# convet x_4 and y_4 to two dimention array

x_4, y_4 = np.meshgrid(x_4, y_4)


# In[26]:


# Generating 3D Plot 

fig = plt.figure(figsize=[16,12])

ax = fig.gca(projection='3d')

ax.set_xlabel('X', fontsize = 20)
ax.set_ylabel('Y', fontsize = 20)
ax.set_zlabel('f(x,y) - Cost', fontsize = 20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap= cm.coolwarm, alpha= 0.4)


plt.show()


# # Partial Derivatives & Symbolic Computation
# # $$\frac{\partial f}{\partial x} = \frac{2x \ln(3) \cdot 3^{-x^2 - y^2}}{\left( 3^{-x^2 - y^2} + 1 \right) ^2}$$
# 
# # $$\frac{\partial f}{\partial y} = \frac{2y \ln(3) \cdot 3^{-x^2 - y^2}}{\left( 3^{-x^2 - y^2} + 1 \right) ^2}$$

# In[27]:


a,b = symbols('x,y')

print('Our cost function f(x,y) is:', f(a,b))
print ('partial derivatives wrt x is:', diff(f(a,b),a))
print('Value of f(x,y) at x =1.8 y=1;0 ', f(a,b).evalf(subs={a:1.8, b:1.0}))


print('', diff(f(a,b),a).evalf(subs={a:1.8, b:1.0}))



# In[ ]:





# # Batch Gradiet Descent with SymPy 

# In[28]:


# setup 

multiplier = 0.1 
max_iter  = 500
params  = np.array([1.8, 1.0]) # intial guess

for n in range(max_iter):
    gradient_x = diff(f(a,b),a).evalf(subs={a:params[0], b:params[1]})
    gradient_y = diff(f(a,b),b).evalf(subs={a:params[0], b:params[1]})
    gradients = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradients
    
 #Result 

print('Values in gradient array', gradients)
print('Minimum occurs at x value of:', params[0])
print('Minimum occur at y value of', params[1] )
print('The cost is:', f(params[0],params[1]))
    
    
    


# In[29]:


# Partial derivative function example 4 

def fpx(x,y):
    r = 3**(-x**2 - y**2)
    return 2*x*log(3)*r/ (r + 1)**2

def fpy(x,y):
    r = 3**(-x**2 - y**2)
    return 2*y*log(3)*r/ (r + 1)**2


# In[30]:


# setup 

multiplier = 0.1 
max_iter  = 500
params  = np.array([1.8, 1.0]) # intial guess

for n in range(max_iter):
    gradient_x = fpx(params[0], params[1])
    gradient_y = fpy(params[0], params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradients
    
 #Result 

print('Values in gradient array', gradients)
print('Minimum occurs at x value of:', params[0])
print('Minimum occur at y value of', params[1] )
print('The cost is:', f(params[0],params[1]))
    
    
    


# # Graphing 3D gradient Descent & Adv Numpy Arrays

# In[58]:


# Generating 3D Plot 

fig = plt.figure(figsize=[16,12])

ax = fig.gca(projection='3d')

ax.set_xlabel('X', fontsize = 20)
ax.set_ylabel('Y', fontsize = 20)
ax.set_zlabel('f(x,y) - Cost', fontsize = 20)

ax.plot_surface(x_4, y_4, f(x_4, y_4), cmap= cm.coolwarm, alpha= 0.4)
ax.scatter(values_array[:,0], values_array[:,1], f(values_array[:,0],values_array[:,1]), s=50, color= 'red')


plt.show()


# In[56]:


# setup 

multiplier = 0.1 
max_iter  = 500
params  = np.array([1.8, 1.0]) # intial guess
values_array = params.reshape(1,2)


for n in range(max_iter):
    gradient_x = fpx(params[0], params[1])
    gradient_y = fpy(params[0], params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradients
    values_array = np.append(values_array, params.reshape(1,2), axis=0 )
    
 #Result 

print('Values in gradient array', gradients)
print('Minimum occurs at x value of:', params[0])
print('Minimum occur at y value of', params[1] )
print('The cost is:', f(params[0],params[1]))
    
    
    


# In[57]:



kirk = np.array([['captain', 'guitar']])
print(kirk.shape)

hs_band = np.array([['black thought','mc' ], ['questlove', 'drums']])

print('hs_band[0][1]', hs_band[0][1])


the_roots = np.append(arr=hs_band, values=kirk, axis=0)

print(the_roots)


# In[ ]:





# # Working with Data &  a Real Cost Function 
# ## Mean Squared Error: a cost function for regression problems 
# ###  $$RSS = \sum_{i=1}^n \big(y^{(i)} - h_\theta x^{(i)}\big)^2 $$
# ###  $$MSE = \frac{1}{n} \sum{i=1}^{n} \big(y^{(i)} - h_\theta x^{(i)}\big)^2 $$
# 

# In[32]:


# make sample data 

x_5 = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y_5 = np.array([1.7, 2.4,3.5,3.0,6.1,9.4,8.2]).reshape(7,1)


# In[35]:


regr = LinearRegression()
regr.fit(x_5, y_5)

print('Theta 0:', regr.intercept_[0])
print('Theta 1:', regr.coef_[0][0])


# In[38]:


plt.scatter(x_5,y_5, s=50)
plt.plot(x_5, regr.predict(x_5), color='orange', linewidth=3)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()


# In[60]:


# y_hat = theta0 + thetal*x

y_hat = 0.847535148603 + 1.22272646378*x_5
print('Est values y_hat are: \n', y_hat)
print('In comparison, the actual y values are \n', y_5)


# In[61]:


def mse(y, y_hat):
    #mse_calc = 1/7 * sum((y - y_hat)**2)
    #mse_calc = (1/y.size) * sum((y - y_hat)**2)
    mse_calc = np.average((y - y_hat)**2, axis=0)
    return mse_calc


# # 3D Plot for the MSE Cost Function 
# ## make data for thetas
# 

# In[66]:


nr_thetas = 200
th_0 = np.linspace(start= -1, stop=3, num=nr_thetas)
th_1 = np.linspace(start= -1, stop=3, num=nr_thetas)


plot_t0, plot_t1 = np.meshgrid(th_0,th_1)
plot_t0


# # Calc MSE using nested for loops 

# In[67]:


plot_cost = np.zeros((nr_thetas, nr_thetas))  

for i in range(nr_thetas):
    for j in range(nr_thetas):
        y_hat = plot_t0[i][j] + plot_t1[i][j]*x_5
        plot_cost[i][j] = mse(y_5, y_hat)
        

print('Shape of plot_t0', plot_t0.shape)
print('Shape of plot_t0', plot_t1.shape)
print('Shape of plot_t0', plot_cost.shape)


# In[68]:




fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)

ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.hot)
plt.show()


# In[69]:


print('min value of plot_cost', plot_cost.min)

ij_min = np.unravel_index(indices=plot_cost.argmin(), dims=plot_cost.shape)
print('Min occurs at (i,j):', ij_min)
print('Min MSE for Theta 0 at plot_t0[111][91]', plot_t0[111][91])
print('Min MSE for Theta 1 at plot_t1[111][91]', plot_t1[111][91])


# ## Partial Derivatives of MSE w.r.t. $\theta_0$ and $\theta_1$
# 
# ## $$\frac{\partial MSE}{\partial \theta_0} = - \frac{2}{n} \sum_{i=1}^{n} \big( y^{(i)} - \theta_0 - \theta_1 x^{(i)} \big)$$
# 
# ## $$\frac{\partial MSE}{\partial \theta_1} = - \frac{2}{n} \sum_{i=1}^{n} \big( y^{(i)} - \theta_0 - \theta_1 x^{(i)} \big) \big( x^{(i)} \big)$$
# 

# ## MSE & Gradient Descent

# In[79]:


# x values, y values, array of theta parameter (theta0 at index 0 and Theta1 at index 1)

def grad(x,y, thetas):
    n = y.size
    
    theta0_slope = (-2/n) * sum(y - thetas[0] - thetas[0]*x)
    theta1_slope = (-2/n) * sum(y - thetas[0] - thetas[0]*x)
    
    return np.concatenate((theta0_slope, theta1_slope), axis=0)


# In[84]:


multiplier = 0.01
thetas = np.array([2.9, 2.9])

#collect data points for scatter plot 
plot_vals = thetas.reshape(1,2)
mse_vals = mse(y_5, thetas[0] + thetas[1]*x_5)

for i in range(1000):
    thetas = thetas - multiplier * grad(x_5, y_5, thetas)
    
    # Append the new values to our numpy arrays
    plot_vals = np.concatenate((plot_vals, thetas.reshape(1,2)), axis=0)
    mse_vals = np.append(arr=mse_vals, values=  mse(y_5, thetas[0] + thetas[1]*x_5))
    
print('Min occurs at Theta 0:', thetas[0])
print('Min occurs at Theta 1:', thetas[1])
print('Mse is :', mse(y_5, thetas[0] + thetas[1]*x_5))


# In[87]:




fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)

ax.scatter(plot_vals[:,0],plot_vals[:,1] , mse_vals, s=80, color = 'black')
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.rainbow, alpha= 0.4)
plt.show()


# In[ ]:




