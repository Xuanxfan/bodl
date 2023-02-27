import model as mo
import os
from matplotlib import pyplot as plt

#creat train data



#load model
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

covModel = mo.Cov_model()
history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))