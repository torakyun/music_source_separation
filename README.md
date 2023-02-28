# 生成的最適化に基づく音楽音源分離の検証

日本語 | [English](docs/en/README.md)

## 音楽音源分離の概要と現状
音楽音源分離は、一度合成された楽曲を、個々の楽音に分解する技術のことを指す。
楽音の個別操作が可能になることで、
音楽配信サービス上で高度な機能を実現することにつながる。
実際に、音楽音源分離は、リミックスやカラオケ音源の作成、
音楽情報検索、音楽教育といった機能への応用が試みられてきた。
しかし、現在の技術では、他の音源からの分離漏れや、
対象の音源自体の欠損といった問題を完全に消すことはできていない。
また、音の一部に雑音が生じてしまうことがあり、
これらの要素が、主観評価を下げる要因になる。

<img width="800" src="https://user-images.githubusercontent.com/67317828/217169110-5bfc0204-bf78-4a7c-9a9f-8bb44e5d6c31.gif">
  
## 生成的最適化に基づく音楽音源分離
音楽音源分離は、過去10年間に集中的に研究されており、
深層学習の登場によって飛躍的な進歩を遂げた。
深層学習による音楽音源分離は、主に2つのアプローチによって発展してきた。
周波数領域におけるマスキングアプローチと、
時間領域における End-to-End アプローチである。
後者のアプローチでは、
抽出した特徴量から時間波形への直接的な再構成を行う。
このような構造は、音声合成分野でも一般的に用いられてきた。
しかし、生成的な構造を取り入れたモデルが提案されている一方で、
これらのモデルに対する最適化は音声合成分野のものとは異なる。
従来の最適化では、時間領域上で波形損失の最小化が行われてきた。
波形損失は、時間波形の各時刻において、教師との差分を取ることで求められる。
つまり、同時刻における単一の信号が比較される。
それに対して、音声合成分野の最適化では、系列信号の時間変化を考慮する。
STFT損失の最小化を行うと、音の周波数特性を学習することができる。
人間の聴覚系も周波数解析を行っており、人間の聴覚に基づいた方法といえる。
また、敵対的損失の最小化を行うと、音の本物らしさを学習することができる。
敵対的損失の最小化は、出力系列が本物の音である確率の最大化に基づいており、
本物らしい生成を可能にする。
そこで本研究では、生成的最適化に基づく音楽音源分離を提案する。
波形損失に加えて、STFT 損失と敵対的損失を同時に最小化する。
波形としての近さだけでなく、音色の違いがより良く学習されることで、
分離精度と音質汚染の主観評価向上が期待できる。

<img width="800" alt="architecture3" src="https://user-images.githubusercontent.com/67317828/217165468-09999921-727f-4427-999d-d29c7e9862b8.png">

## 生成的最適化の検証結果

* **データセット - [MUSDB-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)**

    音楽音源分離用データセットでは最大規模のもの。
    データ拡張を施して使用した。

* **生成器 - [Hybrid Demucs](https://arxiv.org/pdf/2111.03600.pdf)**

    時間領域の End-to-End モデル Demucs を、
    ハイブリッド領域に拡張したもの。

* **識別器 - [StyleMelGAN](https://arxiv.org/pdf/2011.01557.pdf) の識別器**

    軽量かつ高品質な生成を実現。

* **波形損失 - 平均絶対誤差 （Mean Absolute Error:MAE）**

    Hybrid Demucs で用いられている損失関数。

* **STFT損失 - スペクトログラムの差分**

    ```（リニアスケール＋パワースケール）×３つの解像度``` で求める。
    [Parallel WaveGAN](https://arxiv.org/pdf/1910.11480.pdf) を参照。

* **敵対的損失 - 二乗誤差**

    [Least Squares GAN](https://arxiv.org/pdf/1611.04076.pdf) を参照。

* **主観指標 - 平均オピニオン評点（Mean Opinion Score: MOS）**

    各カテゴリへの投票率を評点で重み付けしたもの。（今回の被験者は38名）
    <img width="600" alt="mos_item" src="https://user-images.githubusercontent.com/67317828/221777752-251a920e-d3b4-49a0-ac2d-cbf9872f522f.png">

* **客観指標 - [音源対歪比（Source-to-Distortion Ratio: SDR）](https://hal.inria.fr/inria-00544230/document)**

    信号比較により正解からの歪みの程度を計る。
    手軽で統一的な評価のために一般的に用いられてきたが、
    人間の聴覚とはあまり相関を示さないという報告もされており、
    妥当性については疑問が残る。

（複数スケール・複数解像度の）STFT損失を加えて最適化を行うと、
分離精度と音質汚染の主観評価を、全体的に向上させることが確認できた。
実際に、対数メルスペクトログラムを描画することで、
STFT損失が寄与していることが見てとれた。
ベースの音質汚染のみ若干低下しているが、
この原因は、存在しないはずの高周波成分を除去できたことにあると考えている。
高周波成分は、音波の細かい概形を形作る成分である。
波形の辻褄合わせを行っていた成分が除去されたことで、
波形としての整合性に若干乱れが生じ，音質を低下させたと考えられる。
また、さらに敵対的損失を加えて最適化を行うと、
分離精度を低下させる結果となったが、ベースの音質汚染は改善された。
分離精度を低下させた原因は、
敵対的損失を計算する識別器が過学習をしてしまったことにあると思われる。
対策としては、識別器の軽量化や、過学習を抑える技術の導入などが考えられる。
敵対的損失を学習する際の不安定さを解消することが今後の課題である。

<img width="800" alt="mos" src="https://user-images.githubusercontent.com/67317828/221762798-6f628f2f-f2c5-4d20-b70d-242bf8e42203.png">
<img width="800" alt="sdr" src="https://user-images.githubusercontent.com/67317828/221762357-ec2ab102-bb3c-460b-8165-cc3dde6eb93a.png">

音楽は人間に聴かれることを想定して制作されているため、
音楽音源分離システムの主観評価を向上させることは重要である。
本研究で実装した生成的最適化に基づく音楽音源分離は、
多くの面で従来手法を上回る主観評価結果を得た。
今後も最適化の見直しが行われ、さらなる品質改善に結びつけていくことが望まれる。
しかし、主観評価が必要とされる一方で、
実際には客観評価のみが行われることが一般的である。
今後は、主観評価のための明確な基準を設けることや、
プラットフォームを作成することなどにより、
積極的に主観評価を試すことができる体制を整えることも必要である。

# 使い方

[こちら](docs/setup.md)を参照してください。

# ライセンス

[こちら](LICENSE)を参照してください。
本システムは、[Hybrid Demucs の実装](https://github.com/torakyun/music_source_separation/tree/64ed2cb029301743b2714b3c8fe930c00945842c)を複製して制作を始めました。
音楽音源分離への多大なる貢献に感謝いたします。
```Trainer```クラスの導入により、
学習、検証、評価を繰り返すシステムの大枠には大幅な変更を加えていますが、
個々のデータ処理、音源分離モデルの実装ファイルは、ほとんどそのまま利用しています。
また、音声合成分野の最適化を取り入れるにあたり、
[kan-hayashi](https://github.com/kan-bayashi/ParallelWaveGAN) さんの実装を参考にしました。
代表的な音声合成モデルの実装例を公開していただき、拝謝します。
