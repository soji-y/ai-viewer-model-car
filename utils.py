import os
import re
import shutil
import datetime
import unicodedata
import time
from pathlib import Path
from collections import OrderedDict

# ランダムシード値を取得
def set_random_seed(cfg, def_seed) -> int:
  seed = def_seed
  if hasattr(cfg, "random_seed"):
    if cfg.random_seed == -1:
      seed = int(time.time())
    else:
      seed = int(cfg.random_seed)

  return seed

# 全角を半角に変換
def zenkaku_to_hankaku(text) -> str:
  return ''.join([unicodedata.normalize('NFKC', char) for char in text])

# 最後のフォルダを取得
def get_last_dir(dir_path, last_time_flg=False, front_name="") -> str:
  if not dir_path or not os.path.isdir(dir_path):
    return None
  
  # フォルダ内のサブフォルダをリストアップ
  sub_dirs = sorted([f.path for f in os.scandir(dir_path) if f.is_dir()])

  if not sub_dirs:
    return None
  
  if front_name != "":
    # アダプター名が設定されているときは、それ以外のフォルダは候補から外す
    sub_dirs = [dir for dir in sub_dirs if os.path.basename(dir).startswith(front_name)]

  # 一先ず最後のフォルダを取得
  get_sub_dir = sub_dirs[-1] if sub_dirs else None

  if last_time_flg:
    # 最後の更新日時のフォルダを取得
    last_time_stamp = 0

    for sub_dir in sub_dirs:
      time_stamp = os.path.getmtime(sub_dir)
      if time_stamp > last_time_stamp:
        last_time_stamp = time_stamp
        get_sub_dir = sub_dir
  else:
    # 数値部分で比較して最も大きいフォルダを取得
    def extract_num(s, p, ret=0):
      search = p.search(s)     
      if search:
        return int(search.groups()[0])
      else:
        return ret
      
    p = re.compile(front_name + r'(\d+)$')
    get_sub_dirs = sorted(
      (s for s in sub_dirs if p.match(os.path.basename(s))),
      key=lambda s: extract_num(os.path.basename(s), p, float('inf'))
    )
          
    if not get_sub_dirs:
      return None
    get_sub_dir = get_sub_dirs[-1]

  if get_sub_dir and os.path.isdir(get_sub_dir):
    return str(Path(get_sub_dir))
  else:
    return None

# 空白行のみ削除
# def remove_blank_lines(text) -> str:
#   lines = text.split('\n')
#   non_blank_lines = [line for line in lines if line.strip() != '']
#   result_text = '\n'.join(non_blank_lines)
#   return result_text

# 現在の日付時刻曜日
def get_datetime_base():
  current_time = datetime.datetime.now()
  weekday  = ["月", "火", "水", "木", "金", "土", "日"][current_time.weekday()]
  date_str = f"{current_time.strftime('%Y/%m/%d')}({weekday}) {current_time.strftime('%H:%M:%S')}"
  return date_str

# 現在の日付時刻曜日(チャット履歴用)
def get_datetime_chat() -> str:
  date_str = f"[Datetime: {get_datetime_base()}]"
  return date_str

# 文字列から日付型を取得
def get_date_object(date_string, date_format):
  try:
    date_object = datetime.datetime.strptime(date_string, date_format)
    return date_object
  except ValueError:
    return None  # 変換できない場合はNone

# 日付型から文字列を取得
def get_date_string(date_object:datetime.datetime, date_format):
  try:
    date_string = date_object.strftime(date_format)
    return date_string
  except ValueError:
    return None  # 変換できない場合はNone

# フォルダ内の特定のファイル名検索
def search_files(directory, search_string):
  matched_files = []
  for _, _, files in os.walk(directory):
    for file in files:
      if search_string in file:
        matched_files.append(file)
  return matched_files

# フォルダ内の特定のフォルダ名検索
def search_dirs(directory, search_string):
  matched_dirs = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if search_string in file:
        matched_dirs.append(dir)
  return matched_dirs

# クラスの項目に設定されたパスを結合する
def path_join(class_1, path_str_1, class_2, path_str_2):
  if hasattr(class_1, path_str_1) and hasattr(class_2, path_str_2):
    path = os.path.join(getattr(class_1, path_str_1), getattr(class_2, path_str_2))
  else:
    path = None

  if not os.path.isdir(path):
    path = None
  return path

# 2つのデータセットから同じデータを省いて追加
def same_data_remove_join(dataset01, dataset02):
  set_1 = list(OrderedDict((frozenset(d.items()), d) for d in dataset01).values())
  set_2 = list(OrderedDict((frozenset(d.items()), d) for d in dataset02).values())

  dataset_data = set_1 + set_2

  return dataset_data
