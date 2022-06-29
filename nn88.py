import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from math import log2, floor

def read_raw_data() :
	with open('input.txt') as file :
	    lines = file.readlines()
	    return lines

def proccess_raw_data(raw_data) :
	data = []
	for line in raw_data :
		data.append(list(map(int, line.split())))
	data = np.array(data, dtype = np.longdouble)
	original_labels = np.copy(data[:, 3])
	data[:, 3] = (data[:, 3] - data[:, 0] - 1) / (4 * np.sqrt(data[:, 0])) + 0.5
	return data, original_labels

def generate_X_Y_sets(data) :
	X = [] ; Y = [] ; n = len(data)
	X = data[:, :-1]
	Y = data[:, -1]
	return X, Y, n

def Model() :
	model = tf.keras.Sequential([
		tfl.Dense(units = 512, activation = 'relu', input_dim = 3),
		tfl.BatchNormalization(),

		tfl.Dense(units = 256, activation = 'relu'),
		tfl.Dropout(0.1),
		tfl.BatchNormalization(),

		tfl.Dense(units = 128, activation = 'relu'),
		tfl.Dropout(0.1),
		tfl.BatchNormalization(),

		tfl.Dense(units = 64, activation = 'relu'),
		tfl.Dropout(0.2),
		tfl.BatchNormalization(),

		tfl.Dense(units = 32, activation = 'relu'),
		tfl.Dropout(0.2),
		tfl.BatchNormalization(),

		tfl.Dense(units = 16, activation = 'relu'),
		tfl.Dropout(0.2),
		tfl.BatchNormalization(),

		tfl.Dense(units = 8, activation = 'relu'),
		tfl.Dropout(0.25),
		tfl.BatchNormalization(),

		tfl.Dense(units = 1, activation = 'sigmoid') ])
	return model

data = read_raw_data()
data, original_labels = proccess_raw_data(data)
X, Y, n = generate_X_Y_sets(data)
print('\n Number of examples: ', n, '\n')
P = 1 + floor(log2(X[0, 0]))

ratio = float(input('\n Enter the value of the ratio: '))
split = floor(ratio * n)
X_train, Y_train = X[:split, :], Y[:split]
X_test, Y_test = X[split:, :], Y[split:]
original_labels_train = original_labels[:split]
original_labels_test = original_labels[split:]

model = Model()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.005), loss = 'msle')
model.fit(X_train, Y_train, epochs = 50, batch_size = 64, shuffle = False)

model.save_weights(str(ratio).replace('.', '1') + 'weights.hdf5') ; print('')
#model.load_weights('weights.hdf5') ; print('')
model.evaluate(X_test, Y_test)

n_test_examples = np.shape(X_test)[0]
predictions = np.reshape(model.predict(X_test), n_test_examples)

Y_test[:] = np.round((Y_test[:] - 0.5) * 4 * np.sqrt(X_test[:, 0]) + X_test[:, 0] + 1)
predictions[:] = np.round((predictions[:] - 0.5) * 4 * np.sqrt(X_test[:, 0]) + X_test[:, 0] + 1)

Delta = np.absolute(Y_test - predictions)
avg_diff = np.sum(Delta) // np.shape(Delta)[0] 

print('\n    sqrt(p) = ', round(np.mean(X_test[:, 0])**0.5))
print('   avg_diff = ', round(avg_diff))
print('\n 2*avg_diff = ', round(2*avg_diff))
print(' 4*avg_diff = ', round(4*avg_diff))

print('\n Delta = \n ')
print(Delta[:18])

print('\n', round( 100 * (Delta < 2*avg_diff).sum() / np.shape(Delta)[0] ), '%')

file = open('./output_nn.txt', 'w')
file.writelines( ('|diff|', '    ', 'p', ' ', 'a', ' ', 'b', ' ', 'ord', ' ', 'est' ) )

for i in range(n_test_examples) :

		file.writelines( ('\n', '  ', str(round(Delta[i])), '   ', str(round(X_test[i][0])), ' ',
			str(round(X_test[i][1])), ' ', str(round(X_test[i][2])), ' ', str(round(Y_test[i])), ' ',
			str(round(predictions[i]) ) ))

