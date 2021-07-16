import pandas as pd
import numpy as np

from dcekit.generative_model import GMR


## ファイルのパスを指定
"""
data_dir:データが存在するパス
"""
data_dir = '..\\data\\'

## ファイルの名前の指定
"""
raw_csvdata:dataset.csvの欠損値があるデータ
"""
raw_csvdata = 'raw_dataset.csv'

## ハイパーパラメータの設定
"""
iterations, max_number_of_components, covariance_types:GMRの設定（このままで特に問題はない）
"""
iterations = 10
max_number_of_components = 20
covariance_types = ['full', 'diag', 'tied', 'spherical']

## GMRによるデータの補完
"""
raw_data:欠損値を含むデータセット
     欠損値がある変数をprintで出力
参考:https://datachemeng.com/iterative_gaussian_mixture_regression/
"""
raw_data = pd.read_csv(data_dir+raw_csvdata, index_col=0)
print('==欠損値の有無の確認', '='*10)
print(raw_data.isna().any(axis=0))

# select nan sample numbers
nan_indexes = np.where(raw_data.isnull().sum(axis=1) > 0)[0]
nan_variables = []
effective_variables = []

for sample_number in nan_indexes:
    nan_variables.append(np.where(raw_data.iloc[sample_number, :].isnull() == True)[0])
    effective_variables.append(np.where(raw_data.iloc[sample_number, :].isnull() == False)[0])

for iteration in range(iterations):
    print('\r', iteration + 1, '/', iterations, end='')
    if iteration == 0:
        x = raw_data.dropna(axis=0)
        autoscaled_x_arranged = (raw_data - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    else:
        x = raw_data.copy()

    # standardize x and y
    autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    x_mean = np.array(x.mean(axis=0))
    x_std = np.array(x.std(axis=0, ddof=1))

    # grid search using BIC
    bic_values = np.empty([max_number_of_components, len(covariance_types)])
    for covariance_type_index, covariance_type in enumerate(covariance_types):
        for number_of_components in range(max_number_of_components):
            model = GMR(n_components=number_of_components + 1, covariance_type=covariance_type)
            model.fit(autoscaled_x)
            bic_values[number_of_components, covariance_type_index] = model.bic(autoscaled_x)

    # set optimal parameters
    optimal_index = np.where(bic_values == bic_values.min())
    optimal_number_of_components = optimal_index[0][0] + 1
    optimal_covariance_type = covariance_types[optimal_index[1][0]]
    #    print(iteration + 1, '/', iterations, ', BIC :', bic_values.min())

    # GMM
    model = GMR(n_components=optimal_number_of_components, covariance_type=optimal_covariance_type)
    model.fit(autoscaled_x)

    # interpolation
    for index, sample_number in enumerate(nan_indexes):
        if iteration == 0:
            mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights_for_x = \
                model.predict(autoscaled_x_arranged.iloc[sample_number:sample_number + 1, effective_variables[index]],
                              effective_variables[index], nan_variables[index])
        else:
            mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights_for_x = \
                model.predict(autoscaled_x.iloc[sample_number:sample_number + 1, effective_variables[index]],
                              effective_variables[index], nan_variables[index])
        interpolated_value = mode_of_estimated_mean[0] * x_std[nan_variables[index]] + x_mean[nan_variables[index]]
        raw_data.iloc[sample_number, nan_variables[index]] = interpolated_value
#        print(interpolated_value)
# save interpolated dataset
raw_data.to_csv(data_dir+'interpolated_dataset.csv')

## GMRによる補間の結果を出力
print('==欠損値の有無の確認（補間後）', '='*10)
print(raw_data.isna().any(axis=0))