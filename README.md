# QuickVC(44100Hz 日本語HuBERT対応版)

このリポジトリは、44100Hzの音声を学習および出力できるように編集した[QuickVC-VoiceConversion](https://github.com/quickvc/QuickVC-VoiceConversion)です。但し、以下の点を変更しております。
- ContentEncoderをWaveNetからAttentionに変更
- HuBERT-softのHiddenUnitsを、[日本語HuBERT](https://huggingface.co/rinna/japanese-hubert-base)12層目768dim特徴量に変更
- MS-iSTFT-VITSのsubband数を4⇛8に変更
- ContentEncoderにF0埋め込みを追加。それに従い、PreprocessにF0抽出処理を追加。


<img src="qvcfinalwhite.png" width="100%">

##  ⚠Work in Progress⚠
学習コードのみ実装。推論や事前学習モデル等は後ほど追加するかもしれません。

## 1. 環境構築

Anacondaによる実行環境構築を想定する。

0. Anacondaで"QuickVC"という名前の仮想環境を作成する。[y]or nを聞かれたら[y]を入力する。
    ```sh
    conda create -n QuickVC python=3.8    
    ```
0. 仮想環境を有効化する。
    ```sh
    conda activate QuickVC
    ```
0. このレポジトリをクローンする（もしくはDownload Zipでダウンロードする）

    ```sh
    git clone https://github.com/tonnetonne814/QuickVC-44100-Ja_HuBERT.git
    cd QuickVC-44100-Ja_HuBERT.git # フォルダへ移動
    ```

0. [https://pytorch.org/](https://pytorch.org/)のURLよりPyTorchをインストールする。
    ```sh
    # OS=Linux, CUDA=11.7 の例
    pip3 install torch torchvision torchaudio
    ```

0. その他、必要なパッケージをインストールする。
    ```sh
    pip install -r requirements.txt 
    ```

## 2. データセットの準備

[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)は配布時の音源が24000Hzの為適さないが、説明のために[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)の学習を想定します。

1. [こちら](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)からJVSコーパスをダウンロード&解凍する。
1. 音源を44100Hz16Bitモノラル音源へと変換する。
1. 解凍したフォルダを、datasetフォルダへ移動し、以下を実行する。
    ```sh
    python3 ./dataset/encode.py --model japanese-hubert-base --f0 harvest 
    ```
    > F0抽出のライブラリは、["dio", "parselmouth", "harvest", "crepe"]から選択可能。適宜変更すること。

    
## 3. [configs](configs)フォルダ内のjsonを編集
主要なパラメータを説明します。必要であれば編集する。
| 分類  | パラメータ名      | 説明                                                      |
|:-----:|:-----------------:|:---------------------------------------------------------:|
| train | log_interval      | 指定ステップ毎にロスを算出し記録する                      |
| train | eval_interval     | 指定ステップ毎にモデル評価を行う                          |
| train | epochs            | 指定ステップ毎にモデル保存を行う                          |
| train | batch_size        | 一度のパラメータ更新に使用する学習データ数                |


## 4. 学習
次のコマンドを入力することで、学習を開始する。YourModelNameは自由に変更して良い。
> ⚠CUDA Out of Memoryのエラーが出た場合には、config.jsonにてbatch_sizeを小さくする。

```sh
python train.py -c configs/quickvc_44100.json -m YourModelName
```

学習経過はターミナルにも表示されるが、tensorboardを用いて確認することで、生成音声の視聴や、スペクトログラム、各ロス遷移を目視で確認することができます。
```sh
tensorboard --logdir logs
```

## 5. 推論 & 事前学習モデル
未実装


## 参考文献
- [QuickVC-VoiceConversion](https://github.com/quickvc/QuickVC-VoiceConversion)
- [MB-ISTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [japanese-hubert-base](https://huggingface.co/rinna/japanese-hubert-base)
- [fish-diffusion](https://github.com/fishaudio/fish-diffusion)
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
- [etrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
