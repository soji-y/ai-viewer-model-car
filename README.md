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
!["フォーム画像"](https://github.com/soji-y/ai-viewer-model-car/blob/master/images/ai_viewer_form.png)

## 使用方法
1. 本リポジトリをクローン
2. 「車判別モデル」を上記パスからダウンロード
3. ダウンロードした「車判別モデル」を解凍
4. 解凍した「Idefics2-8B-Instruct_Car」フォルダごと「models」フォルダ内に配置
5. 必要なライブラリをインストール\
(1) venv等の仮想環境に以下のライブラリをインストール\
(2) pytorch等のインストール
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
(3) クローンしたフォルダに遷移し「requirements.txt」のライブラリをインストール
```
pip3 install -r requirements.txt
```
7. 「AI Viewer」の実行\
```
python ai_viewer.py
```
8. 解凍した「車判別モデル」をフォームにドラッグアンドドロップ
9. 任意の車画像を「入力画像ウィンドウ」にドラッグアンドドロップ
10. 入力テキストボックスに任意の質問文を入力して「送信」ボタン押下
