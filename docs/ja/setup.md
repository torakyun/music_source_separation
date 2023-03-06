# 使い方

日本語 | [English](../en/setup.md)

## 必須要件

- [Python](https://www.python.org/) 3.7
- [pip](https://pip.pypa.io/)
- (on Linux) libsndfile1 パッケージ（[soundfile](https://github.com/bastibe/python-soundfile)で必要）
- [MUSDB18-HQ](https://zenodo.org/record/3338373) データセット（または [MUSDB18](https://zenodo.org/record/1117372) データセット）

## インストール

1. ```git clone``` でリモートリポジトリをローカル環境に複製
   ```bash
   $ git clone https://github.com/torakyun/music_source_separation.git
   ```

2. ```pip``` を用いて ```requirements.txt``` からパッケージをインストール
   ```bash
   $ cd mss
   $ pip install -r requirements.txt
   ```

3. [PyTorch](https://pytorch.org/) 1.10.0 をインストール（インストール方法はリンク先を参照）
   - 使用するマシンの GPU（CUDA）環境に合うバージョンのもの

## ディレクトリ構成

```sh
./
   conf/  # ハイパーパラメータ管理
      dataset/  # データセット
      device/  # デバイス
      loss/  # 損失
      model/  # モデル
         generator/  # 生成器
         discriminator/  # 識別器
      optimizer/  # 最適化アルゴリズム
         generator/  # 生成器
         discriminator/  # 識別器
      config.yaml  # 実験設定
   docs/  # ドキュメント
      en/  # 英語版
         README.md  # README
         setup.md  # 環境構築マニュアル
      ja/  # 日本語版
         setup.md  # 環境構築マニュアル
   mss/  # 音楽音源分離の実装
      layers/  # ニューラルネットワーク層の実装
         __init__.py  # モジュールトップ
         pqmf.py  # 擬似直交ミラーフィルタ（PQMF）
      losses/  # 損失関数の実装
         __init__.py  # モジュールトップ
         adversarial_loss.py  # 敵対的損失・識別器損失
         cac_loss.py  # CAC損失
         feat_match_loss.py  # 特徴一致損失
         mel_loss.py  # メルスペクトログラム損失
         mfcc_loss.py  # MFCC損失
         stft_loss.py  # STFT損失
      models/  # モデルの実装
         __init__.py  # モジュールトップ
         demucs_v2.py  # Demucs（v2）
         demucs.py  # Demucs
         hdemucs.py  # Hybrid Demucs
         hifigan.py  # HiFi-GAN の識別器
         melgan.py  # MelGAN の識別器
         parallel_wavegan.py  # Parallel WaveGAN の識別器
         hifigan.py  # StyleMelGAN の識別器
         tasnet.py  # Conv-Tasnet
      optimizers/  # 最適化アルゴリズムの実装
         __init__.py  # モジュールトップ
         radam.py  # RADAM
   (musdb18hq/)  # MUSDB18-HQ データセット
   (out/)  # 出力
      (checkpoints/)  # 学習経過の保存
      (valids/)  # エポックごとの検証結果
      (evals/)  # エポックごとの評価結果
      (mlruns/)  # MLflowのログ出力
      (models/)  # 学習済みモデルの保存
   .gitignore  # Git管理除外設定
   LICENSE  # ライセンス
   make_mos_set.py  # 主観指標MOSのための評価セット作成
   mos.py  # 主観指標MOSの結果集計
   README.md  # README
   requirements.txt  # Python パッケージ要件（pip）
   run.py  # マルチGPUを用いた学習の実行
   sdr.py  # 客観指標SDRの結果集計
   separate.py  # 学習済みモデルを用いた分離音の予測
```

## コマンド例

### モデルの学習（シングルGPU）

1. （推奨）`config.yaml` に存在しないハイパーパラメータの指定は、`conf` ディレクトリ内の該当ファイルを複製して行う。

2. コマンドラインからハイパーパラメータを指定する。（例：標本化周波数2.4kHz、生成器<音源分離モデル> `hdemucs`）
   ```bash
   $ cd mss
   $ python3 -m mss [-dataset.samplerate=24000] [-model/generator=hdemucs]
   ```

3. MLflowのGUIを起動してログを監視する。
   ```bash
   $ cd mlruns/..
   $ mlflow ui
   ```

4. （推奨）```valid_interval``` にエポック間隔を設定しておくと、その間隔ごとに検証で得られた分離音が `valids` ディレクトリ内に保存される。

- （推奨）GPUが複数個ある時には、異なる条件でシングルGPU学習を並列させて行う。
- （推奨）時間的または計算資源の面で効率良く学習するために、データとモデルの軽量化を行う。
- （注意）GPUがメモリが足りない時には、```batch_divide``` でミニバッチを分割して少しずつ学習を行う。

### モデルの学習（マルチGPU）

`mss` モジュールの実行から、`run.py` ファイルの実行に変更する。
```bash
$ cd mss
$ python3 run.py [-dataset.samplerate=24000] [-model/generator=hdemucs]
```

### 学習済みモデルを主観指標MOSで評価

1. 1人分20個のデータからなる評価セットを作成する。
   ```bash
   $ cd mss
   $ python3 make_mos_set.py -n [複数のモデル名] -r [評価セットの個数]
   ```
   コマンドを実行すると、以下の手順で作成される。  
      - MUSDB18の `test` から、1曲につき8秒の無作為抽出を行う。（すべての音源が存在する区間）
      - 20個を無作為に取り出し、音源とモデル（＋正解）を無作為に割り当てる。

2. `HumanEvaluation.ipynb` を編集する。
   - 評価項目を設定する。
   - Google Drive に配置した評価セットへの id を編集。

3. 被験者に `HumanEvaluation.ipynb` の Colaboratory へのリンクを公開し、ブラウザ上で評価してもらう。
   - 評価結果は、csv ファイルとしてダウンロードされる。

4. 評価結果を集計する。
   ```bash
   $ cd mss
   $ python3 mos.py -f [csvファイルが格納されたディレクトリへのパス]
   ```

### 学習済みモデルを客観指標SDRで評価

```bash
$ cd mss
$ python3 sdr.py -n [複数のモデル名]
```

- （注意）メモリが足りない場合は、モデルを1つずつ評価する。

### 任意の楽曲に対して学習済みモデルで音楽音源分離を実行

```bash
$ cd mss
$ python3 separate.py [複数の楽曲] -n [モデル名]
```

- デフォルトでは wav ファイルで書き出される。  
オプション ```--mp3``` で mp3 ファイルに変更できる。
