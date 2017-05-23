import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-1,1,200)
np.random.shuffle(x)
y = 0.5 * x + 2 + 0.01*np.random.normal(0,0.005,(200,))

train_x = x[:160]
train_y = y[:160]

test_x = x[160:]
test_y = y[160:]

model = Sequential()
model.add(Dense(units = 1, input_dim=1))

model.compile(loss="mse",optimizer="sgd")

print("train =====")
for step in range(501):
    cost = model.train_on_batch(train_x,train_y)
    if(step % 50 == 0):
        print("train cost = ", cost)


print("test =====")

cost = model.evaluate(test_x,test_y,batch_size=40)
print("test cost = ", cost )
w,b = model.layers[0].get_weights()

print("W = ",w)
print("b = ", b)
plt.scatter(test_x,test_y)
Y_pred = model.predict(test_x)
plt.plot(test_x,Y_pred)
plt.show()