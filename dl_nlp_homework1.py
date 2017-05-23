import numpy as np
import math

# Task 2.2(a) Training
w_list = [[-1,1]]  #memorize all w value after each pair of train value
x_list = [[-1.28,0.09],[0.17,0.39],[1.36,0.46],[-0.51,-0.32]]
y = [0,1,1,0]

#algorithm
for i in range(len(x_list)):
    x = np.array(x_list[i])
    w = np.array(w_list[i])
    xw = np.dot(x,w)
    sig = 1/math.exp(-1 * xw)
    deri_sig = math.exp(-xw)/math.pow((1+math.exp(-xw)),2)
    w = w - ((w - sig - y[i])*deri_sig * x)
    w_list.append(w)

for i in range(len(w_list)):
    print("w[",i,"] = ",w_list[i])


#Task 2.2(b) Evaluation
w_new = [np.array(w_list[0]),w_list[4]]
l=0
x_li = [[-0.5,-1],[0.75,0.25]]
y_li = [0,1]
x = np.array(x_li)
for i in range(2):
    for j in range(2):
        x = np.array(x_li[j])
        w = w_new[i]
        xw = np.dot(x,w)
        temp = math.pow((1/math.exp(-1 * xw)-y[j]),2)
        l = l+temp
print("\n")
print("l = ",l)


#Task 2.3 Decision Boundary and Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = [-1.28,0.17,1.36,-0.51]
x2 = [0.09,0.39,0.46,-0.32]
y=[0,1,1,0]
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax.scatter(x1, x2, y)
plt.show()