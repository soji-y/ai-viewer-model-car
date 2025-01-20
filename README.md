## モデル
### ■車判別モデル
https://drive.google.com/drive/folders/1U-O0aMDx8fvX3Sgih1Qyqp4MVYoM8WFH
- **Idefics2-8B-Instruct_Car.zip**

## 推奨環境
GPU：Geforce RTX4080 (GPUメモリ16GB以上)

## 動作確認済環境
- Intel(R) Core(TM) i9-14900K 3.20 GHz
- RAM 64 GB
- Geforce RTX4090
- Windows11
- CUDA 12.1

## 動作画面


## 使用方法
1. 本リポジトリをクローン
2. 「車判別モデル」を上記パスからダウンロード
3. ダウンロードした「車判別モデル」を任意の場所に解凍
4. 必要なライブラリのインストール\
(1) pytorchのインストール\
(2) 「requirements.txt」のインストール
5. 「AI Viewer」の実行\
python ai_viewer.py
6. 解凍した「車判別モデル」をフォームにドラッグアンドドロップ
7. 任意の車画像を「入力画像ウィンドウ」にドラッグアンドドロップ
8. 入力プロンプトに任意の質問文を入力して「送信」ボタン押下
