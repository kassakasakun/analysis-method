**解析用スクリプトの説明**\n
interpolate_data_igmr.py がデータ補間用のスクリプト
regression_model.py が解析用のスクリプト

**ディレクトリ構成**
home
    -code
        -regression.py
        -interpolate_data_igmr.py
    -data
        -raw_dataset.csv
        -statistics_data.csv
        (-interpolate_dataset.csv)
    -saves
        -csv
        -image

3つのディレクトリで構成されることを想定しています

・code
    Pythonコードを入れておくディレクトリ
・data
    csvデータを入れておくディレクトリ
    raw_dataset.csv
        dataset.csvのうち解析に使うデータのみを抜き出したcsvファイル
        1行目->Lot番号
        2行目->目的変数（粘度）
        3行目以降->説明変数
    statistics_data.csv
        プロセスデータの統計量データ
        1行目->Lot番号
        2行目以降->統計量データ
    interpolate_dataset.csv
        raw_dataset.csvをinterpolate_data_igmr.pyによって補間したもの
        1行目->Lot番号
        2行目->目的変数（粘度）
        3行目以降->説明変数
・saves
    解析結果を保存するためディレクトリ
    csv
        保存ファイルの内csvのもの
    image
        保存ファイルの内画像のもの

**実行環境**
Python3.xで実行してください
必要となるライブラリ
・pandas
・numpy
・matplotlib
・sikit-learn
・lightgbm
・xgboost
・boruta
・dcekit

**出力ファイルの説明**
fet_importance_*.csv:変数重要度
estimated_y_*.csv:目的変数yの推定値
scores_*.csv:評価指標
*.png:y-yプロット

/withoutstatic:統計量なし
/withstatic:統計量あり
/withstatic_boruta*:統計量あり+Borutaによる変数選択

**使い方**
1.ディレクトリを構成する
    上記のディレクトリ構成に従ってディレクトリやファイルを整理する
2.データの補完
    interpolate_data_igmr.pyをエディタで開く
    「ファイルの名前の指定」の「raw_csvdata」の変数にcsvのファイル名を入力する（変更がある場合）
    スクリプトを実行するとdataディレクトリ内に「interpolate_dataset.csv」が生成されるはずなので確認する
3.解析
    regression_model.pyをエディタで開く
    「ファイルの名前の指定」の「interpolate_csvdata」「statistics_csvdata」の変数に
    csvのファイル名を入力する（変更がある場合）
    「解析方法の設定」のところに記載されている説明に従い解析方法を設定する
    スクリプトを実行するとsavesディレクトリに解析結果が保存されている
