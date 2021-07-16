import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


import lightgbm
import xgboost

from boruta import BorutaPy

## ファイルのパスを指定
"""
data_dir:データが存在するパス
save_dir:出力結果を保存するディレクトリのパス
"""
data_dir = '..\\data\\'
save_dir = '..\\saves\\'

## ファイルの名前の指定
"""
interpolate_csvdata:dataset.csvの欠損値を補間したデータ
statistics_data:プロセスデータの統計量をまとめたデータ
"""
interpolate_csvdata = 'interpolate_dataset.csv'
statistics_csvdata = 'statistics_data.csv'

## 解析方法の設定
"""
use_statics:統計量データを使うかどうか
     使う場合は「True」使わない場合は「False」
use_boruta:Borutaの変数選択を行うかどうか
     使う場合は「True」使わない場合は「False」
p:borutaの閾値
     None:borutaを使わない場合はNoneにしてください
     80,90,100:使いたい閾値に設定
"""
use_statics=False
use_boruta=False
p=None
#p=80
#p=90
#p=100


## ハイパーパラメータの設定
"""
model_names:解析モデルの設定
random_forest_variables_rates:Randomforestで用いる変数の割合
"""
model_names = ['rf', 'gbdt', 'lgb', 'xgb']
random_forest_variables_rates = np.arange(1, 10, dtype=float) / 10


## データの設定
interpolate_data = pd.read_csv(data_dir+interpolate_csvdata, index_col=0)
x_data = interpolate_data.iloc[:, 1:]
y_data = interpolate_data.iloc[:, 0]

## 統計量データの処理
"""
統計量データの読み込み
95%以上が同じ値の変数を削除
"""
if use_statics:
    static_data = pd.read_csv(data_dir+statistics_csvdata, index_col=0)
    drop_list = []
    for i in range(static_data.shape[1]):
        same_data_p = static_data.iloc[:, i].value_counts().iloc[0] / static_data.shape[0]
        if same_data_p > 0.95:
            drop_list.append(static_data.columns[i])
    print('==drop valiables more than 95% same values : ', len(drop_list), '', '='*10)
    x_data = pd.concat([x_data, static_data.drop(columns=drop_list).fillna(0)], axis=1)

## Borutaの変数選択
"""
設定したpに基づいてBorutaを実行
"""
if use_boruta:
    print('==Boruta p:', p, '='*10)
    rf_boruta = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf_boruta, n_estimators='auto', verbose=2, random_state=1, perc=p)
    feat_selector.fit(x_data.values, y_data.values.ravel())
    selected = feat_selector.support_
    print('選択された特徴量の数: %d' % np.sum(selected))
    remain_list = x_data.columns[np.where(selected == True)[0]]
    print(remain_list)
    print('ranking :', feat_selector.ranking_)
    x_data = x_data[remain_list]

## 標準化（オートスケーリング）を行う関数
def autoscaling(d, d_train):
    return (d - d_train.mean(axis=0)) / (d_train.std(axis=0, ddof=1) + 1e-10)

## スケールを戻す為の関数
def rescaling(estimated_d, d_train):
    return estimated_d * (d_train.std(axis=0, ddof=1) + 1e-10) + d_train.mean(axis=0)

## RandomForest回帰モデルの関数
class RF_regression:
    """
    input:
    inner_x,inner_y:ダブルクロスバリデーションの内側データ
    outer_x,outer_y:ダブルクロスバリデーションの外側データ
    rf_variables_rates:変数割合

    output:
    regression_model.feature_importances_:変数重要度
    est_y_test:外側データの予測結果
    """
    def __init__(self, inner_x, inner_y):
        self.x = inner_x
        self.y = inner_y
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x, self.y, train_size=0.8, shuffle=False)
        self.autoscaled_x_train = autoscaling(self.x_train, self.x_train)
        self.autoscaled_y_train = autoscaling(self.y_train, self.y_train)
        self.autoscaled_x_valid = autoscaling(self.x_valid, self.x_train)
    
    def fit(self, rf_variables_rates):
        rmse_oob_all = list()
        for i, random_forest_x_variables_rate in enumerate(rf_variables_rates):
            RandomForestResult = RandomForestRegressor(
                n_estimators=500,
                max_features=int(max(math.ceil(self.autoscaled_x_train.shape[1] * random_forest_x_variables_rate), 1)),
                oob_score=True
                )
            
            RandomForestResult.fit(self.autoscaled_x_train, self.autoscaled_y_train)
            estimated_y_in_cv = RandomForestResult.oob_prediction_
            estimated_y_in_cv = rescaling(estimated_y_in_cv, self.y_train)
            
            rmse_oob_all.append((sum((self.y_train - estimated_y_in_cv) ** 2) / len(self.y_train)) ** 0.5)
        optimal_random_forest_x_variables_rate = rf_variables_rates[np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
        # print(f'\noptimal variable rate : {optimal_random_forest_x_variables_rate}')
        self.regression_model = RandomForestRegressor(
            n_estimators=500,
            max_features=int(max(math.ceil(self.autoscaled_x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)),
            oob_score=True
            )
        self.regression_model.fit(autoscaling(self.x, self.x_train), autoscaling(self.y, self.y_train))
        return self.regression_model.feature_importances_
    
    def predict(self, outer_x, outer_y):
        self.x_test = outer_x
        self.y_test = outer_y
        self.autoscaled_x_test = autoscaling(self.x_test, self.x_train).values
        
        est_y_test = self.regression_model.predict(self.autoscaled_x_test.reshape([1, -1]))
        est_y_test = rescaling(est_y_test, self.y_train)
        
        return est_y_test

## GBDT回帰モデルの関数
class GBDT_regression:
    """
    input:
    inner_x,inner_y:ダブルクロスバリデーションの内側データ
    outer_x,outer_y:ダブルクロスバリデーションの外側データ

    output:
    regression_model.feature_importances_:変数重要度
    est_y_test:外側データの予測結果
    """
    def __init__(self, inner_x, inner_y):
        self.x_train = inner_x
        self.y_train = inner_y
        self.autoscaled_x_train = autoscaling(self.x_train, self.x_train)
        self.autoscaled_y_train = autoscaling(self.y_train, self.y_train)
    
    def fit(self):
        self.regression_model = GradientBoostingRegressor()
        self.regression_model.fit(self.autoscaled_x_train, self.autoscaled_y_train)
        return self.regression_model.feature_importances_
    
    def predict(self, outer_x, outer_y):
        self.x_test = outer_x
        self.y_test = outer_y
        self.autoscaled_x_test = autoscaling(self.x_test, self.x_train).values
        
        est_y_test = self.regression_model.predict(self.autoscaled_x_test.reshape([1, -1]))
        est_y_test = rescaling(est_y_test, self.y_train)
        
        return est_y_test

## XGBoost回帰モデルの関数
class XGB_regression:
    """
    input:
    inner_x,inner_y:ダブルクロスバリデーションの内側データ
    outer_x,outer_y:ダブルクロスバリデーションの外側データ

    output:
    regression_model.feature_importances_:変数重要度
    est_y_test:外側データの予測結果
    """
    def __init__(self, inner_x, inner_y):
        self.x_train = inner_x
        self.y_train = inner_y
        self.autoscaled_x_train = autoscaling(self.x_train, self.x_train)
        self.autoscaled_y_train = autoscaling(self.y_train, self.y_train)
    
    def fit(self):
        self.regression_model = xgboost.XGBRegressor()
        self.regression_model.fit(self.autoscaled_x_train, self.autoscaled_y_train)
        return self.regression_model.feature_importances_
    
    def predict(self, outer_x, outer_y):
        self.x_test = outer_x
        self.y_test = outer_y
        self.autoscaled_x_test = autoscaling(self.x_test, self.x_train).values
        
        est_y_test = self.regression_model.predict(self.autoscaled_x_test.reshape([1, -1]))
        est_y_test = rescaling(est_y_test, self.y_train)
        
        return est_y_test

## LightGBM回帰モデルの関数
class LGB_regression:
    """
    input:
    inner_x,inner_y:ダブルクロスバリデーションの内側データ
    outer_x,outer_y:ダブルクロスバリデーションの外側データ

    output:
    regression_model.feature_importances_:変数重要度
    est_y_test:外側データの予測結果
    """
    def __init__(self, inner_x, inner_y):
        self.x_train = inner_x
        self.y_train = inner_y
        self.autoscaled_x_train = autoscaling(self.x_train, self.x_train)
        self.autoscaled_y_train = autoscaling(self.y_train, self.y_train)
    
    def fit(self):
        self.regression_model = lightgbm.LGBMRegressor()
        self.regression_model.fit(self.autoscaled_x_train, self.autoscaled_y_train)
        return self.regression_model.feature_importances_
    
    def predict(self, outer_x, outer_y):
        self.x_test = outer_x
        self.y_test = outer_y
        self.autoscaled_x_test = autoscaling(self.x_test, self.x_train).values
        
        est_y_test = self.regression_model.predict(self.autoscaled_x_test.reshape([1, -1]))
        est_y_test = rescaling(est_y_test, self.y_train)
        
        return est_y_test

## 予測結果の評価およびプロット図作成の関数
def evaluation(raw_y, estimated_y, model_name):
    """
    input:
    raw_y:実測値のyデータ
    estimated_y:推定値のyデータ
    model_name:手法の名前

    output:
    evaluation_list:評価指標のリスト[r2, MAE, RMSE]
    """
    raw_y_array = raw_y.values
    estimated_y_array = estimated_y.reshape((estimated_y.shape[0]))

    ## y-yプロット図の作成
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(5, 5))
    plt.scatter(raw_y, estimated_y_array)
    y_max = np.max(np.array([raw_y_array, estimated_y_array]))
    y_min = np.min(np.array([raw_y_array, estimated_y_array]))
    plt.plot(
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 
        'k-'
        )
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('actual y')
    plt.ylabel('estimated y')
    if use_statics == False:
        plt.savefig(f'{save_dir}image\\{model_name}_withoutstatic.png', bbox_inches='tight')
    elif use_boruta:
        plt.savefig(f'{save_dir}image\\{model_name}_withstatic_boruta{p}.png', bbox_inches='tight')
    else:
        plt.savefig(f'{save_dir}image\\{model_name}_withstatic.png', bbox_inches='tight')
    plt.show()

    # r2dcv, RMSEdcv, MAEdcvの計算
    r2 = round(metrics.r2_score(raw_y, estimated_y_array), 5)
    mae = round(metrics.mean_absolute_error(raw_y, estimated_y_array), 5)
    rmse = round(np.sqrt(metrics.mean_squared_error(raw_y, estimated_y_array)), 5)

    evaluation_list = [r2, mae, rmse]
    print(f'\n{model_name} score:')
    print(f'r2 : {r2}')
    print(f'MAE : {mae}')
    print(f'RMSE : {rmse}')

    return evaluation_list

## 解析
"""
evaluation_df:評価指標をまとめるための表
regression_model_dict:回帰分析手法をまとめた辞書型データ
estimated_y_all_pd:予測結果を保存するための表
"""
evaluation_df = pd.DataFrame(index=model_names, columns=['r2', 'MAE', 'RMSE'])
regression_model_dict = {
    model_names[0] : RF_regression,
    model_names[1] : GBDT_regression,
    model_names[2] : XGB_regression,
    model_names[3] : LGB_regression
    }
estimated_y_all_pd = pd.DataFrame(index=x_data.index, columns=model_names)

## モデルごとでモデル構築->予測を行う
for model_num in model_names:
    print('==', model_num, '='*10)
    regression_model = regression_model_dict[model_num]
    estimated_y_all = np.zeros_like(y_data)
    feature_importance = pd.DataFrame(index=x_data.index, columns=x_data.columns)

    ## DCVの外側のサンプル分繰り返し計算を行う
    for i in range(interpolate_data.shape[0]):
        print(f'\r{i+1}/{interpolate_data.shape[0]}', end='')
        inner_x = x_data.drop(index=x_data.index[i])
        outer_x = x_data.iloc[i, :]
        inner_y = y_data.drop(index=y_data.index[i])
        outer_y = y_data.iloc[i]
        
        model = regression_model(inner_x, inner_y)
        if model_num == 'rf':
            feature_importance.iloc[i, :] = model.fit(random_forest_variables_rates)
        else:
            feature_importance.iloc[i, :] = model.fit()
        estimated_y_all[i] = model.predict(outer_x, outer_y)
    
    ## 変数重要度をcsvに保存
    if use_statics == False:
        feature_importance.to_csv(f'{save_dir}csv\\fet_importance_{model_num}_withoutstatic.csv')
    elif use_boruta:
        feature_importance.to_csv(f'{save_dir}csv\\fet_importance_{model_num}_withstatic_boruta{p}.csv')
    else:
        feature_importance.to_csv(f'{save_dir}csv\\fet_importance_{model_num}_withstatic.csv')
    
    ## 予測結果を評価
    evaluation_df.loc[model_num, :] = evaluation(y_data, estimated_y_all, model_num)
    estimated_y_all_pd[model_num] = estimated_y_all

## 推定値yをcsvに保存
if use_statics == False:
    estimated_y_all_pd.to_csv(f'{save_dir}csv\\estimated_y_withoutstatic.csv')
elif use_boruta:
    estimated_y_all_pd.to_csv(f'{save_dir}csv\\estimated_y_withstatic_boruta{p}.csv')
else:
    estimated_y_all_pd.to_csv(f'{save_dir}csv\\estimated_y_withstatic.csv')
print('==evaluation score', '='*10)
print(evaluation_df)

## 評価指標をcsvに保存
if use_statics == False:
    evaluation_df.to_csv(f'{save_dir}csv\\scores_withoutstatic.csv')
elif use_boruta:
    evaluation_df.to_csv(f'{save_dir}csv\\scores_withstatic_boruta{p}.csv')
else:
    evaluation_df.to_csv(f'{save_dir}csv\\scores_withstatic.csv')
