# 入力画像ウィンドウ
import os
import re
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES
import pillow_heif

# pillow-heifプラグインを有効化
pillow_heif.register_heif_opener()

# 画像読込ダイアログの拡張子
IMAGE_FILE_TYPE = [("画像ファイル", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.heic"), ("すべてのファイル", "*.*")]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic")

FORM_TITLE = "入力画像ウィンドウ"
MENU_TOPMOST = "最前面表示"
MENU_IMAGE_LOAD = "画像を読み込み"
MENU_IMAGE_CLEAR = "画像をクリア"
MENU_CLOSE = "閉じる"

# 入力画像ウィンドウクラス
class PictureWindow(tk.Toplevel):
  def __init__(self, app):
    super().__init__()
    
    self.owner = app

    self.title(FORM_TITLE)
    self.geometry("500x500")

    # キャプチャ時間の計測フラグ
    self.measure_flg = False

    # 画像を常時出力フラグ
    self.img_save_flg = False

    # フォームの背景色(録画時の枠)
    self.back_color = "red"
    if hasattr(self.owner, "actors"):
      if len(self.owner.actors)>0 and self.owner.actors[0]:
        self.back_color = self.owner.actors[0].chat_color

    self.cap_frame_x = (10, 10)
    self.cap_frame_y = (10, 10)

    # キャンバス(Canvas)を作成して表示
    self.bg_color = "gray40" # 背景色
    self.canvas = tk.Canvas(self, bg=self.bg_color)
    self.canvas.pack(fill=tk.BOTH, expand=True)
    self.canvas.config(bd=0, highlightthickness=0)  # キャンバスの枠を非表示

    # コンテキストメニューの作成
    self.context_menu = tk.Menu(self, tearoff=0)    
    self.var_topmost = tk.BooleanVar() # 最前面

    self.context_menu.add_checkbutton(label=MENU_TOPMOST, variable=self.var_topmost, command=self.menu_topmost)
    self.context_menu.add_separator()
    self.context_menu.add_command(label=MENU_IMAGE_LOAD, command=self.menu_load)
    self.context_menu.add_command(label=MENU_IMAGE_CLEAR, command=self.menu_clear)
    self.context_menu.add_separator()
    self.context_menu.add_command(label=MENU_CLOSE, command=self.on_closing)

    # 画像の参照を保持する変数を作成
    self.image_path_list = []
    self.image_dic_list = []

    self.image_disp = None # 画面に表示する画像
    self.image_tk = None # リサイズした後の画像
    
    # 最前面表示
    self.var_topmost.set(True)
    if self.var_topmost.get():
      self.attributes('-topmost', self.var_topmost.get())

    # 選択画像Index
    self.list_idx = -1

    # 画像のループ
    self.var_loop = True

    # ドラッグ＆ドロップの設定
    self.drop_target_register(DND_FILES)
    self.dnd_bind('<<Drop>>', self.on_drop)

    # 左クリックイベントをバインド
    self.bind("<Button-1>", self.on_mouse_click)

    # 右クリックイベントをバインド
    self.bind("<Button-3>", self.show_context_menu)    
    
    # マウスホイールイベント
    self.bind("<MouseWheel>", self.on_mouse_wheel)

    # ウィンドウサイズ変更時のイベントをバインド
    self.bind("<Configure>", self.on_resize)

    # サブウィンドウが閉じられるときのイベントを設定
    self.protocol("WM_DELETE_WINDOW", self.on_closing) 

  # 画像を切り替える
  def change_image(self):
    if len(self.image_path_list) > 0:
      if len(self.image_path_list) <= self.list_idx + 1:
        self.list_idx = 0
      else:
        self.list_idx += 1
      try:
        path = self.image_path_list[self.list_idx]
        img = Image.open(path).convert("RGB")
        self.image_dic_list[0] = {"data": img, "path": path} 
        self.display_image(img_data=img)
      except Exception as e:
        print(f"Error: displaying image: {e}")

  # メニューを表示(マウス右ボタン：クリック)
  def show_context_menu(self, event):
    # ポップアップメニューを表示する
    self.context_menu.post(event.x_root, event.y_root)

  # メニュー：最前面表示
  def menu_topmost(self):
    self.attributes('-topmost', self.var_topmost.get())

  # メニュー：画像を読込
  def menu_load(self):
    image_file = filedialog.askopenfile(title="画像を読込", filetypes=IMAGE_FILE_TYPE)
    if image_file:
      try:
        self.image_path_list.clear()
        self.image_dic_list.clear()
        
        image_path = image_file.name
        self.image_path_list.append(str(Path(image_path)))
        self.list_idx = 0
        self.display_image(image_path)

      except Exception as e:
        print(f"Error: Failed to Load Image. [{image_path}]")

  # メニュー：画像をクリア
  def menu_clear(self):
    self.image_path_list.clear()
    self.image_dic_list.clear()
    self.image_disp = None
    self.image_tk = None
    self.set_form_title()
    self.list_idx = -1
    self.canvas.delete("all")

  # マウス左ボタン：クリック
  def on_mouse_click(self, event):
    if self.list_idx < 0: return

    if self.list_idx < len(self.image_path_list) - 1:
      self.list_idx += 1
    else:
      if self.var_loop:
        self.list_idx = 0

    path = self.image_path_list[self.list_idx]
    self.display_image(file_path=path)

  # マウスホイール：変更
  def on_mouse_wheel(self, event):
    if self.list_idx < 0: return

    if event.num == 4 or event.delta > 0:
      if self.list_idx > 0:
        self.list_idx -= 1
      else:
        if self.var_loop:
          self.list_idx = len(self.image_path_list) - 1
  
    elif event.num == 5 or event.delta < 0:
      if self.list_idx < len(self.image_path_list) - 1:
        self.list_idx += 1
      else:
        if self.var_loop:
          self.list_idx = 0

    path = self.image_path_list[self.list_idx]
    self.display_image(file_path=path)

  # ドラッグドロップ
  def on_drop(self, event):
    file_paths = re.findall(r'\{[^}]+\}|[^\s]+', event.data)
    file_paths = [path[1:-1] if path.startswith("{") and path.endswith("}") else path for path in file_paths]

    self.menu_clear()
    self.list_idx = -1

    for drop_path in file_paths:
      drop_path = str(Path(drop_path))
      if os.path.isdir(drop_path): # フォルダ読み込み
        files = [f for f in os.listdir(drop_path) if f.lower().endswith(IMAGE_EXT)]
        get_path_list = [os.path.join(drop_path, f) for f in files]

        # 複数画像入力
        for path in get_path_list:
          path = str(Path(path))
          try:
            self.image_path_list.append(path)
            # 画像を表示する
            if len(self.image_dic_list) == 0:
              img = Image.open(path).convert("RGB")
              self.image_dic_list.append({"data": img, "path": path})
          except Exception as e:
            print(f"Error: displaying image: {e}")

      elif os.path.isfile(drop_path): # ファイル読み込み

        if drop_path.lower().endswith(IMAGE_EXT):
          try:
            self.image_path_list.append(str(Path(drop_path)))
            # 画像を表示する          
            if len(self.image_dic_list) == 0:
              img = self.display_image(drop_path)
            self.list_idx = 0
          except Exception as e:
            print(f"Error: displaying image: {e}")
      else:
        print("Warning: Not Input Image Format.")

      title_str = FORM_TITLE

      if len(self.image_path_list) > 0:
        title_str += f" [{1}/{self.image_path_list}]"

      if len(self.image_dic_list) > 0:
        self.list_idx = 0
        self.display_image(img_data=self.image_dic_list[0]["data"])

      self.title()

  # 画面リサイズ
  def on_resize(self, event):
    # ウィンドウサイズが変更されたときに画像を再サイズ
    self.resize_image()

  # サブウィンドウが閉じられる直前に呼び出される関数
  def on_closing(self):
    self.image_dic_list.clear()
    self.image_path_list.clear()
    
    if hasattr(self.owner, "win_picture"):
      self.owner.var_view_pict.set(False)
      self.owner.win_picture = None

    self.destroy()

  # タイトルを変更
  def set_form_title(self):
    title_str = FORM_TITLE

    path_list = self.image_path_list
    idx = self.list_idx
    if len(path_list) > 0:

      title_str += f" [{idx+1}/{len(path_list)}]"
      title_str += f" - {os.path.basename(path_list[idx])}"

    self.title(title_str)

  # 画面に画像を表示
  def display_image(self, file_path=None, img_data=None):
    try:
      if img_data:
        image = img_data
      else:
        file_path = str(Path(file_path))
        image = Image.open(file_path).convert("RGB")
        if len(self.image_dic_list) == 0:
          self.image_dic_list.append({"data": image, "path": file_path})
        else:
          self.image_dic_list[0] = {"data": image, "path": file_path} 

      self.image_disp = image

      self.resize_image()
      
    except Exception as e:
      print(f"Error loading image: {e}")

    self.set_form_title()
    return image

  # 画像リサイズ
  def resize_image(self):
    if len(self.image_dic_list) == 0:
      return

    image = self.image_disp

    if image:
      try:
        # ウィンドウのサイズに合わせて画像をリサイズ
        window_width = self.winfo_width()
        window_height = self.winfo_height()

        # 縦横比を保ったリサイズ
        aspect_ratio = image.width / image.height
        if window_width / window_height > aspect_ratio:
          new_width = window_height * aspect_ratio
          new_height = window_height
        else:
          new_width = window_width
          new_height = window_width / aspect_ratio

        # 画像をリサイズ
        resized_image = image.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(resized_image, master=self.canvas)

        # キャンバス上に画像を表示（中央に配置）
        self.canvas.delete("all")
        self.canvas.create_image((window_width - new_width) // 2, (window_height - new_height) // 2, anchor=tk.NW, image=self.image_tk)
      except Exception as e:
        print(f"Error resizing image: {e}")
