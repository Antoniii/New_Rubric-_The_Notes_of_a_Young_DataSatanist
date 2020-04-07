# Пример регрессии: предсказание цен на дома

# загрузка набора данных
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# подготовка данных
# нормализация
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# конструирование сети
# определение модели
from keras import models
from keras import layers

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model


# перекрёстная проверка по К блокам
import numpy as np

k = 4
num_val_sample = len(train_data) // k
'''
num_epochs = 100
all_scores = []

for i in range(k):
	print('processing fold #', i)
	# подготовка проверочных данных из блока k
	val_data = train_data[i * num_val_sample: (i + 1) * num_val_sample]
	val_target = train_targets[i * num_val_sample: (i + 1) * num_val_sample]

	# подготовка обучающих данных из остальных блоков
	partial_train_data = np.concatenate([train_data[:i * num_val_sample], train_data[(i + 1) * num_val_sample:]], axis=0)
	partial_train_targets = np.concatenate([train_targets[:i * num_val_sample], train_targets[(i + 1) * num_val_sample:]], axis=0)

	# конструирование модели Keras
	model = build_model()

	# обучение в режиме без иывода сообщений (verbose=0)
	model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)

	# оценка модели по проверочным данным
	val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
	all_scores.append(val_mae)
'''
# print(all_scores)
# print(np.mean(all_scores))

# сохранение оценки проверки перед каждым прогоном
num_epochs = 500
all_mae_histories = []

for i in range(k):
	print('processing fold #', i)
	# подготовка проверочных данных из блока k
	val_data = train_data[i * num_val_sample: (i + 1) * num_val_sample]
	val_target = train_targets[i * num_val_sample: (i + 1) * num_val_sample]

	# подготовка обучающих данных из остальных блоков
	partial_train_data = np.concatenate([train_data[:i * num_val_sample], train_data[(i + 1) * num_val_sample:]], axis=0)
	partial_train_targets = np.concatenate([train_targets[:i * num_val_sample], train_targets[(i + 1) * num_val_sample:]], axis=0)

	# конструирование модели Keras
	model = build_model()

	# обучение в режиме без иывода сообщений (verbose=0)
	history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_target), epochs=num_epochs, batch_size=1, verbose=0)

	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)

# создание истории последовательных средних оценок проверки по К блокам
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)] 

# формирование графика с оценками проверок
import matplotlib.pyplot as plt 
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('validation_MAE.png')

# формирование графика с оценками проверок за исключением первых 10 замеров
def smooth_curve(points, factor=0.9):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points
	return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('short_validation_MAE.png')

# обучение окончательно версии модели
# получить новую скомпилированную модель
model = build_model()

# обучить на всем объёме обучающих данных
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score) # средняя ошибка 2.9696527462379607


# предсказание на новых данных
predict_list = model.predict(test_data)
with open('ex3.txt', 'w') as f:
	for i in range(len(predict_list)):
		f.write(str(predict_list[i]))
		f.write('\n')