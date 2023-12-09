import time
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from moexalgo import Market, Ticker

stocks = Market('stocks')

# Акции SBER
sber = Ticker('SBER')
# Акции OZON
ozon = Ticker('OZON')
# Акции TATNEFT
tatneft = Ticker('TATN')
# Акции YNDX
yandex = Ticker('YNDX')
# Акции GAZP
gazprom = Ticker('GAZP')

# Получение текущего локального времени
current_time = time.localtime()
day = current_time.tm_mday

company = [sber, ozon, tatneft, yandex, gazprom]

company_to_save = ["sber", "ozon", "tatneft", "yandex", "gazprom"]

columns = ['Stock', 'Date', 'Predicted_Close', 'Signal_rol']

result_dataframe = pd.DataFrame(columns=columns)

for i in range(0, len(company)):
  # Сборка данных со всех датафреймов
  trade = company[i].tradestats(date=f'2023-12-0{day-1}', till_date=f'2023-12-0{day}')
  trade.rename(columns={'systime':'systime_trade'}, inplace=True)
  order = company[i].orderstats(date=f'2023-12-0{day-1}', till_date=f'2023-12-0{day}')
  order.rename(columns={'systime':'systime_order'}, inplace=True)
  obs = company[i].obstats(date=f'2023-12-0{day-1}', till_date=f'2023-12-0{day}')
  obs.rename(columns={'val_b':'val_b_obs', 'val_s':'val_s_obs', 'vol_b':'vol_b_obs', 'vol_s':'vol_s_obs',
                      'systime':'systime_obs'}, inplace=True)

  data = pd.merge(trade, order, on=['ticker','tradedate', 'tradetime'])
  data = pd.merge(data, obs, on=['ticker','tradedate', 'tradetime'])

  # Оценка важности признаков
  # RandomForestRegressor metod
  exclude_columns = ['ticker', 'tradedate','tradetime', 'systime_trade',
                    'systime_order', 'systime_obs']
  data_ = data.copy()
  data_['tradedate'] = pd.to_datetime(data_['tradedate'])
  data_['tradetime'] = pd.to_datetime(data_['tradetime'], format='%H:%M:%S').dt.time
  data_['datetime'] = pd.to_datetime(data_['tradedate'].astype(str) + ' ' + data_['tradetime'].astype(str))
  data_['datetime'] = data_['datetime'] + pd.to_timedelta('7 days')
  data_ = data_.drop(columns=['tradedate', 'tradetime'])
  data = data.drop(columns=exclude_columns)
  target_column = 'pr_close'
  X = data.drop(columns=[target_column]).fillna(0)
  y = data[target_column]
  rf_model = RandomForestRegressor(random_state=42)
  rf_model.fit(X, y)
  feature_importance_df = rf_model.feature_importances_
  feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

  # Метод одномерного анализа вариации (ANOVA):
  f_scores, p_values = f_classif(X, y)
  anova_importance_df = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores, 'P-Value': p_values})
  anova_importance_df = anova_importance_df.sort_values(by='F-Score', ascending=False)

  # Метод взаимной информации:
  mi_scores = mutual_info_regression(X, y)
  mi_importance_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
  mi_importance_df = mi_importance_df.sort_values(by='MI Score', ascending=False)

  # Пермутационные важности:
  perm_importance = permutation_importance(rf_model, X, y, n_repeats=30, random_state=42)
  perm_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': perm_importance.importances_mean})
  perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)

  # Отбираем наиболее значимые признаки
  fi = list(feature_importance_df[0:10]['Feature'].values)
  ai = list(anova_importance_df[0:10]['Feature'].values)
  mi = list(mi_importance_df[0:10]['Feature'].values)
  pi = list(perm_importance_df[0:10]['Feature'].values)

  result = list(set(fi) & set(ai) & set(mi) & set(pi))

  # Передаем их в модель для предсказывания

  # Создание целевой переменной (target) - прогноз цены закрытия на следующем временном интервале
  data_['target'] = data_['pr_close'].shift(-1)

  # Удаление последней строки, так как у нее нет целевой переменной
  candles_data = data_[:-1]

  # Выбор признаков (features)
  features = candles_data[result]

  # Нормализация данных
  scaler = MinMaxScaler()
  scaled_features = scaler.fit_transform(features)

  # Создание последовательности данных для LSTM
  def create_sequences(data_, sequence_length):
      sequences = []
      targets = []
      for j in range(len(data_) - sequence_length):
          sequences.append(data_[j:j+sequence_length])
          targets.append(data_[j+sequence_length, 0])  # Используем только первый столбец
      return np.array(sequences), np.array(targets)

  sequence_length = 10  # длина последовательности
  X, y = create_sequences(scaled_features, sequence_length)

  # Разделение данных на обучающий и тестовый наборы
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Создание и обучение модели LSTM
  model = Sequential()
  model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

  # Получение прогнозов на будущий месяц
  future_features = data_[result]
  scaled_future_features = scaler.transform(future_features)

  # Преобразование в последовательности
  X_future, y_future = create_sequences(scaled_future_features, sequence_length)

  # Получение прогнозов
  future_predictions = model.predict(X_future)

  # Инвертирование нормализации для визуализации результатов
  scaled_inv = scaler.inverse_transform(scaled_future_features)

  # Преобразование даты в формат datetime
  # data_['tradetime'] = pd.to_datetime(data_['tradetime'])
  data_.set_index('datetime', inplace=True)

  # Создание DataFrame для прогнозов
  future_predictions_df = pd.DataFrame({
      'Stock': data_['ticker'],
      'Date': data_.index[:len(scaled_inv)],
      'Predicted_Close': scaled_inv[:, 0]
  })

  # Применение стратегии скользящих средних:
  future_predictions_df['short_ma'] = future_predictions_df['Predicted_Close'].rolling(window=20).mean()
  future_predictions_df['long_ma'] = future_predictions_df['Predicted_Close'].rolling(window=50).mean()

  # Сигналы покупки и продажи
  future_predictions_df["Signal_rol"] = 0
  future_predictions_df.loc[future_predictions_df['short_ma'] > future_predictions_df['long_ma'], 'Signal_rol'] = 1
  future_predictions_df.loc[future_predictions_df['short_ma'] < future_predictions_df['long_ma'], 'Signal_rol'] = -1

  future_predictions_df['Previous_Signal_Rol'] = future_predictions_df['Signal_rol'].shift(1)

  # Создаем все возможные пары сигналов
  all_signal_pairs = pd.merge(future_predictions_df, future_predictions_df, how='cross')

  # Отфильтруем пары, где сигнал изменился (купить -> продать)
  entry_exit_pairs = all_signal_pairs[(all_signal_pairs['Signal_rol_x'] == 0) & (all_signal_pairs['Signal_rol_y'] == 1)]

  # Рассчитаем доходность для каждой пары
  entry_exit_pairs['Returns'] = entry_exit_pairs['Predicted_Close_y'] - entry_exit_pairs['Predicted_Close_x']

  # Найдем пару с максимальной доходностью
  best_trade = entry_exit_pairs.loc[entry_exit_pairs['Returns'].idxmax()]

  # Найдем даты для покупки и продажи
  entry_date = best_trade['Date_x']
  exit_date = best_trade['Date_y']

  future_predictions_df['Signal_rol'] = 0

  # Обновление столбца Signal_Rol
  future_predictions_df.loc[future_predictions_df['Date'].isin([entry_date, exit_date]), 'Signal_rol'] = [1, -1]

  future_predictions_df = future_predictions_df.drop(columns=['short_ma', 'long_ma', 'Previous_Signal_Rol'])

  # Индексы, где Signal_rol равен 1 или -1
  indexes_1 = future_predictions_df[future_predictions_df['Signal_rol'] == 1].index
  indexes_minus_1 = future_predictions_df[future_predictions_df['Signal_rol'] == -1].index

  # Всего 30 значений
  total_values = 30

  # Равные интервалы для оставшихся значений
  remaining_indexes = []
  step = len(future_predictions_df) // (total_values - 2)
  for k in range(0, len(future_predictions_df), step):
      remaining_indexes.append(future_predictions_df.index[k])

  # Объединение всех индексов и выбор уникальных
  selected_indexes = set(indexes_1) | set(indexes_minus_1) | set(remaining_indexes)
  
  # print(future_predictions_df)
  # Получение итогового DataFrame
  result_data = future_predictions_df.loc[sorted(list(selected_indexes))]

  result_dataframe = pd.concat([result_dataframe, result_data], ignore_index=True)

result_dataframe.to_csv("result_day.csv")
# result_dataframe
