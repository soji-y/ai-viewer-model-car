import os
import gc
import re
import yaml
import time
import datetime
import random
import argparse
import threading
import traceback
import torch
from pathlib import Path
from typing import Optional
from loguru import logger
from PIL import Image, ImageTk, ImageDraw
from qwen_vl_utils import process_vision_info

import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES

from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, LlamaConfig
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
from peft import PeftModel

from constant import *
from utils import *
from win_picture import PictureWindow

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # ライブラリの重複を許可
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0を使用
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # メモリの断片化を抑制

# カレントディレクトリの設定
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 日付文字列用のパターン
datetime_ptn = re.compile(r"\[Datetime:(.*?)\]")
img_path_ptn = re.compile(r"\[Image:(.*?)\]")


# PrintとLoggerへの出力(Type:0=info,1=error,2=warning)
def print_log(strings:str, no_new_line=False, log_new_line=False, print_no_type=False, log_type=0):
  print_str = strings
  if not print_no_type:
    if log_type == 0:
      print_str = f"Info: {print_str}"
    elif log_type == 1:
      print_str = f"Error: {print_str}"
    elif log_type ==2:
      print_str = f"Warning: {print_str}"

  if no_new_line:
    print(print_str, end='')
  else:
    print(print_str)

  if logger:
    if log_new_line:
      strings = "\n" + strings
    if log_type == 0:
      logger.info(strings)
    elif log_type == 1:
      logger.error(strings)
    elif log_type ==2:
      logger.warning(strings)


# プロパティクラス
class Property():
  def __init__(self, cfg):
    self.cfg = cfg

  def get_property(self, actor=None, att_name:str="", def_value=None):    
    value = def_value
    if property:
      if hasattr(self.cfg, att_name):
        value = getattr(self.cfg, att_name)

    if actor:
      if hasattr(actor, att_name):
        value = getattr(actor, att_name)

    return value


# 設定クラス
class Config:
  def __init__(self, config_path):

    config = None
    with open(config_path, 'r', encoding="utf8") as file:
      config = yaml.safe_load(file)

    if not config:
      return None

    cfg_default = config["default"]
    cfg_settings = config["settings"]
    cfg_form = config["form"]
    cfg_log = config["log"]
      
    self.def_actor = ""
    if cfg_default["actor"] and "path" in cfg_default["actor"]:
        self.def_actor = cfg_default["actor"]["path"]
    cfg_user = cfg_default["user"]
    self.user_name = cfg_user["name"]

    self.def_color = "black"
    self.def_bg_color = "gray80"
    self.other_names = []
    self.other_colors = []
    self.other_bg_colors = []
    if "def_chat_color" in cfg_default:
      self.def_color = cfg_default["def_chat_color"]
    if "def_chat_bk_color" in cfg_default:
      self.def_bg_color = cfg_default["def_chat_bk_color"]
    if "other_speakers" in cfg_default:
      cfg_others = cfg_default["other_speakers"]
      for other in cfg_others:
        self.other_names.append(other["name"])
        self.other_colors.append(other["color"] if other["color"] else self.def_color)
        self.other_bg_colors.append(other["bg_color"] if other["bg_color"] else self.def_bg_color)

    self.no_chat_key = "※" # チャット中止文字(デフォルト)
    self.chat_split_key = "---" # チャット終了文字列(デフォルト)

    # [setting]辞書のキーと値をクラスメンバとして設定
    for key, value in cfg_user.items():
        setattr(self, key, value)

    # デフォルト設定
    if not hasattr(self, "hidden_names"): self.hidden_names = [self.user_name] # 隠す名前リスト
    if not hasattr(self, "chat_color"): self.chat_color = "black" # チャット文字色
    if not hasattr(self, "chat_bg_color"): self.chat_bg_color = "gray80" # チャット背景色

    # [setting]辞書のキーと値をクラスメンバとして設定
    for key, value in cfg_settings.items():
        setattr(self, key, value)

    # ここで設定しておいた方が良いものを設定
    if not hasattr(self, "img_redraw_msec"): self.img_redraw_msec = 100 # メイン画像再描画時間(ミリ秒)
    if not hasattr(self, "chat_hist_flg"): self.chat_hist_flg = True # チャット履歴
    if not hasattr(self, "max_sentences"): self.max_sentences = -1 # チャット履歴なしの文数
    if not hasattr(self, "max_sentences_hist"): self.max_sentences_hist = 2 # チャット履歴ありの文数
    if not hasattr(self, "chat_emotion_model"): self.chat_emotion_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    if not hasattr(self, "multi_chat_flg"): self.multi_chat_flg = False
    if not hasattr(self, "rejected_mem_flg"): self.rejected_mem_flg = False # 拒否結果を保存する有無

    # フォント(初期設定)
    self.font_form = ["meiryo", 10]
    self.font_prof = ["meiryo", 9]
    self.font_chat = ["meiryo", 10]
    self.font_cmd = ["meiryo", 10]

    if "font" in cfg_form:
      cfg_font = cfg_form["font"]
      if "form" in cfg_font:
        self.font_form = cfg_font["form"]
      if "prof" in cfg_font:
        self.font_prof = cfg_font["prof"]
      if "chat" in cfg_font:
        self.font_chat = cfg_font["chat"]
      if "cmd" in cfg_font:
        self.font_cmd = cfg_font["cmd"]

    self.win_char_view_sec = 0.3
    
    # 思考ウインドウの文字色、背景色、フォント
    self.win_thought_lines = 100 # 表示可能な行数
    self.win_thought_bg = "black"
    self.win_thought_fg = "white"
    self.win_thought_font = ["meiryo", 9]
    self.win_thought_font_b = ["meiryo", 12]

    # 追加指示ウインドウの文字色、背景色、フォント
    self.win_instruct_lines = 30 # 表示可能な行数
    self.win_instruct_bg = "black"
    self.win_instruct_fg = "white"
    self.win_instruct_font = ["meiryo", 9]

    # [form]辞書のキーと値をクラスメンバとして設定
    for key, value in cfg_form.items():
        setattr(self, key, value)

    # ログ設定
    self.log_output_flg = False
    if "output_flg" in cfg_log:
      self.log_output_flg = cfg_log["output_flg"]
    if "path" in cfg_log:
      self.log_path = cfg_log["path"]

    # ログ設定
    self.setting_logger()

    if not hasattr(self, "color_txt_chat"): self.color_txt_chat = "gray97"

    # Deviceの設定
    self.setting_device(cfg_settings)

    # メモリ未保存確認フラグ
    self.mem_confirm_flg = self.memory_confirmation_flg if hasattr(self, "memory_confirmation_flg") else False

  # デバイスの設定
  def setting_device(self, cfg_settings):
    if "use_cpu_flg" in cfg_settings:
      if cfg_settings["use_cpu_flg"]:
        self.device = torch.device("cpu")
      else:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_log(f"Torch Device is [{self.device.type}].")
    if torch.cuda.is_available():
      print_log(f"CUDA Device Count is [{torch.cuda.device_count()}]. " + \
                f"CUDA Current Device No. is [{torch.cuda.current_device()}]. " + \
                f"CUDA Device Name is [{', '.join([f'{i}: {torch.cuda.get_device_name(i)}' for i in range(torch.cuda.device_count())])}]. " + \
                f"CUDA Device Capability is [{torch.cuda.get_device_capability()}].")

  # ログファイルの設定
  def setting_logger(self):

    # ログの基本設定
    logger.remove()

    if self.log_output_flg:
      if hasattr(self, "log_path"):
        log_dir = self.log_path
      else:
        log_dir = "./log"

      num = 1
      current_time = datetime.datetime.now()
      log_path_full = str(Path(os.path.join(log_dir, f"log_{current_time.strftime('%Y%m%d')}_{num:04}.txt")))

      # ローテーションを設定
      logger.add(log_path_full, rotation="500 MB", enqueue=True)
      logger.info(f"Log Path is [{log_path_full}]")


# アクター(モデル)クラス
class actor:
  def __init__(self, actor_path, cfg):
    self.cfg = cfg
    self.path = actor_path

    # self.param_categorys = ["emotions", "desires"]
    # self.param_emot_items = ["happiness", "surprise", "sadness", "anger", "shyness"]
    # self.param_desir_items = ["eat", "sleep", "sexual", "knowledge"]

    self.owner = None  # 起動アプリ

    # 他スレッドでチェックされるので早めに設定しておく
    self.model = None # Transformers
    self.tokenizer = None

    self.model_loading_flg = False # モデル読み込み中フラグ
    self.load_lora_flg = False # LoRA読み込みフラグ

    # self.generator = None # cTranslate2
    # self.classifier = None # 感情分析

    # self.eye_model = None # 目のモデル
    # self.eye_tokenizer = None # 目のトークナイザー



    # self.main_thread = None # メインループスレッド
    # self.main_stop_event = None # メインループのストップイベント
    # self.main_action_stop_event = None # メインループ内のチャットのストップイベント
    # self.main_action_thread = None # メインループ内のキャラアクションスレッド

    # self.is_chat_now = False # チャット中フラグ
    # self.is_occur_action = False # アクション発生状態の有無
    # self.do_action = None # アクション内容
    # self.action_ratio = 0.001 # アクションの発生確率
    # self.last_action_time = None # 最後のアクション日時
    # self.action_cool_time = 1800 # アクションのクールタイム(秒)

    # キャラクターコンフィグファイルを読み込む
    self.load_config()

    self.think_history = ["---"] # 思考の履歴

  # コンフィグファイルを読み込む
  def load_config(self, no_param_flg=False):
    try:
      actor_cfg_path = os.path.join(self.path, self.cfg.actor_data)

      if not os.path.isfile(actor_cfg_path):
        print_log(f"Error: Not Fonnd! [{actor_cfg_path}]", 1)
        return
      
      data=None
      # キャラ設定ファイルを読み込む
      with open(actor_cfg_path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)

      # モデルパラメータ
      actor_model = data["model"]
      self.model_enable = True
      if "enable" in actor_model:
        self.model_enable = actor_model["enable"]
      self.model_name = actor_model["name"]
      if "tokenizer_name" in actor_model:
        self.tokenizer_name = actor_model["tokenizer_name"]

      self.model_path = actor_model["path"]
      if "type" in actor_model:
        self.model_type = actor_model["type"]
      if "template" in actor_model:
        self.template = actor_model["template"]
      if "bits" in actor_model:
        self.model_bits = actor_model["bits"]

      self.repetition_penalty = actor_model["repetition_penalty"] if "repetition_penalty" in actor_model else 1.05 # デフォルト
      self.do_sample = actor_model["do_sample"] if "do_sample" in actor_model else True # デフォルト
      self.temperature = actor_model["temperature"] if "temperature" in actor_model else 1.0 # デフォルト
      self.max_new_tokens = actor_model["max_new_tokens"] if "max_new_tokens" in actor_model else 256 # デフォルト
      self.merge_flg = actor_model["merge_flg"] if "merge_flg" in actor_model else True # デフォルト
      self.type_vision = actor_model["type_vision"] if "type_vision" in actor_model else False # デフォルト
      self.input_image_max = actor_model["input_image_max"] if "input_image_max" in actor_model else -1 # デフォルト

      # マージモデルの設定
      # self.merge_enable = False
      # self.conv_ct2 = False # cTranslate2モデルにコンバートするか
      # self.full_merge_path = None
      # self.merge_path = None
      # self.merge_name = None
      # if "merge" in actor_model:
      #   actor_merge = actor_model["merge"]
      #   if "enable" in actor_merge:
      #     self.merge_enable = actor_merge["enable"]
      #   if "conv_ct2" in actor_merge:
      #     self.conv_ct2 = actor_merge["conv_ct2"]
      #   if "path" in actor_merge:
      #     self.merge_path = actor_merge["path"]
      #   if "name" in actor_merge:
      #     self.merge_name = actor_merge["name"]

      # モデルが設定されていなかったとき
      # if not hasattr(self, "model_path"): self.model_path = "model"
      # if not hasattr(self, "model_name"): self.model_name = "fast-"
      # if "ct2_quant_type" in actor_model:
      #   self.ct2_quant_type = actor_model["ct2_quant_type"]
      # else:
      #   self.ct2_quant_type = "int8"

      self.full_model_path = os.path.join(self.path, self.model_path)
      # self.full_merge_path = os.path.join(self.path, self.merge_path) if self.merge_path else self.full_model_path

      # if os.path.isdir(self.full_merge_path) and self.merge_name:
      #   # マージモデルの最終パスを取得
      #   self.full_merge_path_num = get_last_dir(self.full_merge_path, False, self.merge_name)
      # else:
      #   self.full_merge_path_num = None

      # LoRAの設定
      # self.full_fine = False
      self.lora_path = None
      if "lora" in data:
        actor_lora = data["lora"]
        # if "full_fine" in actor_lora:
        #   self.full_fine = actor_lora["full_fine"]
        if "path" in actor_lora:
          self.lora_path = actor_lora["path"]
          self.full_lora_path = os.path.join(self.path, self.lora_path) if self.lora_path != "" else None
        # if "save_name" in actor_lora:
        #   self.lora_save_name = actor_lora["save_name"]
        if "adapter_name" in actor_lora:
          self.lora_adapter_name = actor_lora["adapter_name"]
        if "bf16" in actor_lora:
          self.lora_bf16 = actor_lora["bf16"]
        if "bits" in actor_lora:
          self.lora_bits = actor_lora["bits"]

      actor = data["actor"] # [必須]
      self.name = actor["name"] # [必須]

      # キャラパラメータ(その他)：辞書のキーと値をクラスメンバとして設定
      for key, value in actor.items():
          setattr(self, key, value)

      # 設定ファイルになかったらデフォルト
      if not hasattr(self, "chat_name"): self.chat_name = self.name # チャット表示名
      if not hasattr(self, "chat_color"): self.chat_color = "black" # チャット表示色

      # 好きなもの
      if hasattr(self, "favorite"):
        # [favorite]辞書のキーと値をクラスメンバとして設定
        for key, value in self.favorite.items():
          setattr(self, "favorite_" + key, value)

      # self.mem_path = self.train_path = self.work_path = self.image_path = self.event_path = self.name
      # if "data_dir" in data:
      #   data_dir = data["data_dir"]
      #   if "def" in data_dir:
      #     self.mem_path = self.train_path = self.sum_path = self.work_path = self.image_path = data_dir["def"]
      #   if "memory" in data_dir:
      #     self.mem_path = data_dir["memory"]
      #   if "summary" in data_dir:
      #     self.sum_path = data_dir["summary"]
      #   if "train" in data_dir:
      #     self.train_path = data_dir["train"]
      #   if "work" in data_dir:
      #     self.work_path = data_dir["work"]
      #   if "image" in data_dir:
      #     self.image_path = data_dir["image"]

      # チャット設定
      self.sys_prompt_top = ""
      self.sys_prompt_top_only_flg = False
      self.max_sentences = -1
      self.max_effect = -1
      if "chat" in data:
        date_chat = data["chat"]
        if "sys_prompt_top" in date_chat:
          self.sys_prompt_top = date_chat["sys_prompt_top"]
        if "sys_prompt_top_only_flg" in date_chat:
          self.sys_prompt_top_only_flg = date_chat["sys_prompt_top_only_flg"]
        if "no_sys_flg" in date_chat:
          self.chat_no_sys_flg = date_chat["no_sys_flg"]
        if "max_sentences" in date_chat:
          self.max_sentences = date_chat["max_sentences"]
        if "max_effect" in date_chat:
          self.max_effect = date_chat["max_effect"]

      # 能力を取得
      self.get_ability(data)

      self.def_bubble_chat = False
      self.def_actor_img = False
      self.def_instruct = False
      self.def_picture = False
      self.def_thought = False
      # Settingを取得
      if "window" in data:
        window = data["window"]
        for key, value in window.items():
          setattr(self, key, value)

      # Settingを取得
      if "setting" in data:
        setting = data["setting"]
        for key, value in setting.items():
          setattr(self, key, value)

      self.img_path = actor["img_path"] if "img_path" in actor else None
      self.img_path_full = os.path.join(self.path, self.img_path) if self.img_path else None

      # 学習設定(入力名と出力名を取得)
      self.train_input = []
      self.train_output = []
      if "train" in data:
        if "input" in data["train"]:
          self.train_input = data["train"]["input"]
        if "output" in data["train"]:
          self.train_output = data["train"]["output"]

      if not hasattr(self, "def_cuda_num"): self.def_cuda_num = 0
      if not hasattr(self, "def_sub_cuda_num"): self.def_sub_cuda_num = 0

      if "test" in data:
        if "file" in data["test"]:
          self.test_file = data["test"]["file"]
        if "num" in data["test"]:
          self.test_num = data["test"]["num"]
      if not hasattr(self, "test_file"): self.test_file = None
      if not hasattr(self, "test_num"): self.test_num = 1

      self.main_img = None
      self.model_loading_flg =False

      # ベースシステムプロンプト
      self.base_sys_prompt = self.get_sys_prompt()
      
    except Exception as e:
      print(f"Error: Failed to Load actor Config.")
      print_log(f"Error: {str(e)}", 1)

      traceback.print_exc()
      return False

    return True

  # モデルを読み込み
  def load_model(self, model_name=None, tokenizer_name=None, lora_dir_path=None, lora_only=False, cuda_num=0, stop_event=None):
    if not self.model_enable:
      return
    
    model_flg = True if model_name else False # モデル使用フラグ

    if self.model_loading_flg:
      # モデル読み込み中の時は待機
      while self.model_loading_flg:
        time.sleep(0.1)
      # この時点で読み込みは完了してるので、必ずself.loaded_model_nameが設定されている。
      if not self.model_name is None and self.loaded_model_name == self.model_name:
        # 一つ前に読み込んだモデル名が現在のモデル名と一致していれば終了。
        # 一致していなければ新しいモデルを読み込み。
        return
      
      # if not self.full_merge_path_num is None and self.loaded_model_name == self.full_merge_path_num:
      #   # マージモデルでもチェック
      #   return

    # modelとgeneratorとtokenizerを完全に開放する
    self.release_model()
    time.sleep(0.5)

    self.model_loading_flg = True
    self.model_merged_flg = False # モデルをマージ済みか
    try:
    # if True:
      if not lora_only:
        if not self.model is None:
          self.model = None

        if not model_name:
          model_name = self.model_name
        if not tokenizer_name:
          if hasattr(self, "tokenizer_name"):
            tokenizer_name = self.tokenizer_name
          else:
            tokenizer_name = model_name
        if not lora_dir_path:
          if hasattr(self, "full_lora_path"):
            lora_dir_path = self.full_lora_path

        if os.path.isdir(os.path.join(self.path, model_name)):
          # キャラフォルダ内にモデルが存在した場合は優先
          model_name = os.path.join(self.path, model_name)
        if not tokenizer_name is None and os.path.isdir(tokenizer_name):
          pass
        elif not tokenizer_name is None and os.path.isdir(os.path.join(self.path, tokenizer_name)):
          tokenizer_name = os.path.join(self.path, tokenizer_name)
        else:
          tokenizer_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, encoding="utf-8", do_lower_case=False)
        print(f"Loaded Tokenizer Name! [{tokenizer_name}]")
        if self.tokenizer.pad_token is None:
          print("Tokenizer Pad Token is None.")
          self.tokenizer.pad_token = self.tokenizer.eos_token

        # トークナイザーの語彙サイズを確認
        print(f"Tokenizer Vocab Count: {len(self.tokenizer.get_vocab())}")

        # loraフォルダ内の最新のアダプターを取得(存在しない場合はNone)
        if hasattr(self, "lora_adapter_name"):
          self.last_lora_dir_path = get_last_dir(lora_dir_path, False, self.lora_adapter_name)
        else:
          self.last_lora_dir_path = None

        # if hasattr(self, "model_type") and "GPTQ" in self.model_type:
        #   # GPTQモデルを読み込み
        #   self.model = AutoGPTQForCausalLM.from_quantized(model_name, use_safetensors=True, device_map="auto")
        #   # エンベディング層をリサイズ
        #   self.model.resize_token_embeddings(len(self.tokenizer))
        # else:
        # それ以外
        # ct2_enable = False # cTranslate2のモデルを使用できるか
        # if self.cfg.use_ctranslate2_flg and not model_flg and self.merge_enable:
        #   if not self.full_merge_path_num is None and os.path.isdir(self.full_merge_path_num):
        #     if self.last_lora_dir_path is None:
        #       ct2_enable = True
        #     else:
        #       # LoRAとcTranslateのモデル番号を比較
        #       merge_num = os.path.basename(self.full_merge_path_num).replace(self.merge_name,"")
        #       lora_num = os.path.basename(self.last_lora_dir_path).replace(self.lora_adapter_name,"")

        #       # 「_」以降を取る
        #       if '_' in merge_num: merge_num = merge_num.split('_')[0]
        #       if '_' in lora_num: lora_num = lora_num.split('_')[0]

        #       if not merge_num.isdigit() or not lora_num.isdigit():
        #         # どっちかが数値に変換できないときは比較をあきらめる
        #         ct2_enable = True
        #       else:
        #         if int(merge_num) >= int(lora_num):
        #           # LoRAかCT2か数値が大きい方を優先
        #           ct2_enable = True

        # if ct2_enable:
        #   import ctranslate2

        #   # cTranslate2のモデルを使用(条件が厳しめ)
        #   # self.generator = ctranslate2.Generator(self.full_merge_path_num, device=self.cfg.device.type, device_index=cuda_num) #device=f"cuda:0") #"auto")
        #   self.generator = ctranslate2.Generator(self.full_merge_path_num, device=self.cfg.device.type)

        #   lora_dir_path = None # Generatorの時はLoRAは使用しない
        # else:

        # Llama
        load_in_4bit = False
        load_in_8bit = False
        if self.model_bits == 4:
          load_in_4bit = True
        elif self.model_bits == 8:
          load_in_8bit = True

        if "Qwen2-VL" in model_name:
          self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name,
                                                            torch_dtype=torch.float16,
                                                            use_cache=True,
                                                            device_map="auto",
                                                            low_cpu_mem_usage=True,
                                                            load_in_4bit=load_in_4bit,
                                                            load_in_8bit=load_in_8bit,
                                                            )
          self.processor = AutoProcessor.from_pretrained(model_name)

        elif "idefics2-8b" in model_name:
          from transformers import BitsAndBytesConfig, Idefics2ForConditionalGeneration
          torch_dtype = torch.float16

          bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
            ) 
          self.model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            )
          
          self.processor = AutoProcessor.from_pretrained(model_name)
          self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
          # マージフラグを追加
          self.model.config.is_merged = True

        elif "phi-3.5-vision" in model_name:
          self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
            )

          self.processor = AutoProcessor.from_pretrained(model_name, 
            trust_remote_code=True, 
            num_crops=4
            ) 
        else:
          self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            torch_dtype=torch.float16,
                                                            config=LlamaConfig,
                                                            use_cache=True,
                                                            device_map="auto",
                                                            low_cpu_mem_usage=True,
                                                            load_in_4bit=load_in_4bit,
                                                            load_in_8bit=load_in_8bit,
                                                            )

        # エンベディング層をリサイズ
        self.model.resize_token_embeddings(len(self.tokenizer))

        # マージフラグを追加
        self.model.config.is_merged = False





        if not self.model:
          # print(f"Error: Failed to Load Model!")
          print_log(f"Failed to Load Model!", log_type=1)
          self.model_loading_flg = False
          return

        if not self.model:
          # print(f"Info: Loaded cTransrate2 Model! [{str(Path(self.full_merge_path_num))}]")
          print_log(f"Loaded cTransrate2 Model! [{str(Path(self.full_merge_path_num))}]")
        else:
          # print(f"Info: Loaded Model! [{model_name}]")
          print_log(f"Loaded Model! [{model_name}]")

      # 途中終了
      if stop_event and stop_event.is_set():
        self.model_loading_flg = False
        return

      self.load_lora_flg = False

      # if self.generator:
      #   # cTranslateのときはLoRA読み込みなし
      #   pass
      if not lora_dir_path:
        # LoRAが存在しないときは読み込みなし
        print_log("Without using LoRA.")
      else:
        # Loraあり
        # # loraフォルダ内の最新のアダプターを取得
        if self.last_lora_dir_path:

          if "GPTQ" in self.model_type:
            # 4bit量子化されているときはfloat16に戻す
            self.model = self.model.to(torch.float16)

          # self.model = self.model.to(device=cfg.device)

          self.model = PeftModel.from_pretrained(self.model, self.last_lora_dir_path, torch_dtype=torch.float16, device_map="auto", load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit) #{'': 0})

          # LoRAのマージフラグ(先にeval()をするためここではマージしない)
          self.model.config.is_merged = True

          self.load_lora_flg = True
          print_log(f"Loaded LoRA-Adapter! [{self.last_lora_dir_path}]")

        else:
          # Loraなし
          print_log("Without using LoRA.")

      # モデルを保存
      # self.model.save_pretrained("save_model")

      if self.model:
        # 推論モードに変更
        self.model.eval()

        if self.model.config.is_merged: # and "Qwen2-VL" not in self.model_name:
          if self.merge_flg:
            # LoRAとマージ
            self.model = self.model.merge_and_unload()
            self.model_merged_flg = True

      if not self.model is None:
        self.loaded_model_name = self.model_name # 読み込んだモデル名を保存
      # elif not self.generator is None:
      #   self.loaded_model_name = self.full_merge_path_num # 読み込んだモデル名を保存

      if self.load_lora_flg:
        self.loaded_lora_name = str(Path(self.lora_path)/os.path.basename(self.last_lora_dir_path))
      else:
        self.loaded_lora_name = ""

      self.model_loading_flg = False
    except Exception as e:
      print(f"Error: Failed to Load Model or LoRA.")
      print(f"Error: {str(e)}")
      traceback.print_exc()
      messagebox.showerror(MSG_TTL_ERR, f"モデルの読み込みに失敗しました。\n[{str(e)}]")
      self.loaded_model_name = ""
      self.model_loading_flg = False

    stop_event=None

  # modelとtokenizerを完全に開放する
  def release_model(self):
    if not self.model is None:
      self.model = None
    # if not self.generator is None:
    #   self.generator = None
    if not self.tokenizer is None:
      self.tokenizer = None
    # if not self.eye_model is None:
    #   self.eye_model = None
    # if not self.eye_tokenizer is None:
    #   self.eye_tokenizer = None

    if self.cfg.device.type == "cuda":
      torch.cuda.empty_cache()
    gc.collect()

  # メインキャラ画像を読み込み
  def load_main_img(self, img_path=None):
    self.main_img = None
    try:
      if not img_path:
        if not self.img_path_full is None:
          if os.path.isfile(self.img_path_full):
            img = Image.open(self.img_path_full).convert("RGB")
            self.main_img = img
      elif os.path.isfile(img_path):
        img = Image.open(img_path).convert("RGB")
        self.main_img = img
    except Exception as e:
      print(f"Error: {str(e)}")

  # キャラ能力の設定値を取得
  def get_ability(self, data):
    # self.emotion_flg = False # 感情フラグ
    # self.think_flg = False # 思考フラグ
    # self.think_interval = 0 # 思考感覚(秒)
    # self.think_max = 10 # 思考最大数(-1: 無限)
    # self.sys_param_flg = False # システムプロンプトにパラメーターを追加
    # self.sys_time_flg = False # システムプロンプトに日付を追加
    # self.sys_weather_flg = False # システムプロンプトに天気予報を追加
    # self.chat_think_flg = False # チャット中の思考の有無
    # self.chat_think_num = 0 # チャット中の思考の回数
    # self.chat_think_tokens = 64 # チャット中の思考の最大トークン数
    self.chat_remenb_num = 10 # チャットで記憶できる会話数
    self.action_list = []

  # 画面用のプロフィールを取得
  def get_profile(self):

    prof_lines = []

    # prof_lines.append("【プロフィール】")

    # 名前(カナ)：nameは必須
    prof_line = f"[名前] {self.name}"
    if hasattr(self, "name_kana"):
      if self.name != self.name_kana:
        prof_line += f" ({self.name_kana})"
    prof_lines.append(prof_line)

    # その他備考
    prof_line =""
    if hasattr(self, "remarks"):
      prof_line += f"[備考]\n{self.remarks}"
      prof_lines.append(prof_line)

    str_prof = "\n".join(prof_lines)
    return str_prof

  # ベースのシステムプロンプトを取得
  def get_sys_prompt(self):
    sys_prompt = ""
    
    # システムプロンプトの先頭文字列を先に取得
    if hasattr(self, "sys_prompt_top"):
      sys_prompt = self.sys_prompt_top

    return sys_prompt

  # キャラのパラメーターをアップデートする(追加分のみ)
  # def update_param(self, add_param_dic):
  #   if not add_param_dic: return

  #   categorys = self.param_categorys
  #   for category in categorys:
  #     actor_cate = getattr(self, category)
  #     for item, value in add_param_dic.items():
  #       if value == 0: continue
  #       if isinstance(value, (int, float)):
  #         if hasattr(actor_cate, item):
  #           item_value = getattr(actor_cate, item)
  #           # 値を更新する
  #           change_value = item_value + value
  #           if change_value < 0:
  #             change_value = 0
  #           # ひとまず上限値はなし
  #           if change_value > 100:
  #             change_value = 100

  #           if change_value != item_value:
  #             # 値を更新
  #             setattr(actor_cate, item, change_value)

  # パラメーターファイルを読み込む
  # def load_parameter(self):
  #   if os.path.isfile(self.full_param_file):
  #     # 読み込みモードでutf-8エンコーディングでyamlファイルを開く
  #     with open(self.full_param_file, 'r', encoding='utf-8') as file:
  #       data = yaml.safe_load(file)

  #     categorys = self.param_categorys
  #     for category in categorys:
  #       if category in data:
  #         cate_data = data[category]
  #         if category == categorys[0]:
  #           if "happiness" in cate_data:
  #             self.emotions.happiness = cate_data["happiness"]
  #           if "surprise" in cate_data:
  #             self.emotions.surprise = cate_data["surprise"]
  #           if "sadness" in cate_data:
  #             self.emotions.sadness = cate_data["sadness"]
  #           if "anger" in cate_data:
  #             self.emotions.anger = cate_data["anger"]
  #           if "shyness" in cate_data:
  #             self.emotions.shyness = cate_data["shyness"]

  #         if category == categorys[1]:
  #           if "eat" in cate_data:
  #             self.desires.eat = cate_data["eat"]
  #           if "sleep" in cate_data:
  #             self.desires.sleep = cate_data["sleep"]
  #           if "sexual" in cate_data:
  #             self.desires.sexual = cate_data["sexual"]
  #           if "knowledge" in cate_data:
  #             self.desires.knowledge = cate_data["knowledge"]
  #   else:
  #     print("Info: Parameter File not yet Exist.")

  # # パラメーターファイルに書き込む
  # def save_parameter(self):
  #   if self.full_param_file:
  #     categorys = self.param_categorys
  #     param_data = {}
  #     for category in categorys:
  #       if hasattr(self, category):
  #         param_cate = {}
  #         if category == categorys[0]:
  #           # Emotions
  #           if hasattr(self.emotions, "happiness"):
  #             param_cate["happiness"] = self.emotions.happiness
  #           if hasattr(self.emotions, "surprise"):
  #             param_cate["surprise"] = self.emotions.surprise
  #           if hasattr(self.emotions, "sadness"):
  #             param_cate["sadness"] = self.emotions.sadness
  #           if hasattr(self.emotions, "anger"):
  #             param_cate["anger"] = self.emotions.anger
  #           if hasattr(self.emotions, "shyness"):
  #             param_cate["shyness"] = self.emotions.shyness

  #         if category == categorys[1]:
  #           # Desires
  #           if hasattr(self.desires, "eat"):
  #             param_cate["eat"] = self.desires.eat
  #           if hasattr(self.desires, "sleep"):
  #             param_cate["sleep"] = self.desires.sleep
  #           if hasattr(self.desires, "sexual"):
  #             param_cate["sexual"] = self.desires.sexual
  #           if hasattr(self.desires, "knowledge"):
  #             param_cate["knowledge"] = self.desires.knowledge

  #         param_data[category] = param_cate

  #     # 書き込み
  #     with open(self.full_param_file, 'w', encoding='utf-8') as file:
  #       yaml.safe_dump(param_data, file, default_flow_style=False, sort_keys=False) #, allow_unicode=True)

  # イベントが発生するか
  def occur_event(self):
    # self.is_occur_action = False
    # if random.randint(1, 10) == 1:
    if random.random() < self.action_ratio:
      self.is_occur_action = True


# ◆チャットクラス
class Chat():
  def __init__(self):
    self.history_dic_list:list = []   # フル入力のチャット履歴(リスト)

    self.bef_prompt = "" # 前回の入力プロンプト
    self.bef_response = "" # 前回のレスポンス

    self.print_log = None

  # ★[チャット] 送信メッセージを編集して受け取る
  def get_response_actor(self, actor, send_message, sys_prompt, image_set, video_set, max_tokens, temperature, temp_type, actor_name, user_name, no_chat_key, sent_num, direct_flg=False, no_print=False, name_no_msg_flg=False):

    if not no_print:
      s_time = time.time()

    response = self._get_response(actor, send_message, sys_prompt, image_set, video_set, max_tokens, temperature, temp_type, direct_flg, no_log=no_print, user_name=user_name, no_chat_key=no_chat_key, name_no_msg_flg=name_no_msg_flg)

    if not no_print:
      e_time = time.time()
      time_text = f"{e_time-s_time:.2f}s"
      self.print_log(f"Response Time [{time_text}]", print_no_type=True)

      # そのままを表示
      response_full = '\n'.join(['* ' + s for s in response.split("\n")])
      response_text = f"*** Full Response " + "*"*82 + f"\n{response_full}\n" + "*"*100
      # self.print_log(response_text, log_new_line=True, print_no_type=True)
      print(response_text)
      logger.info("*** Full Response ***\n" + response)

    # if not full_response_flg:
    #   # システムな文字の前をカット
    #   response = self._cut_system_strings(response, actor_name=actor_name)
    # チャットEXのときは後処理しない
    # if not ex_flg:
    #   # 改行をカットして結合
    #   response = self._cut_newline_strings(response)
    #   if not full_response_flg:
    #     # 変換不可能文字を置換
    #     response = self._conv_unconv_chr(response)
    #     # 繰り返し文(過去のチャット履歴と同一)をカット
    #     if not sys_prompt: sys_prompt = ""
    #     response = self._cut_send_message(response, sys_prompt + "\n" + send_message).strip()
    #     # 名前をカット(必須)
    #     name_list = [actor.name] + [user_name] + FIRST_PERSONS + LINES_VALUE
    #     # 名前をカット(任意)
    #     if hasattr(actor, "chat_name"): name_list += [actor.chat_name]
    #     if hasattr(actor, "name_kana"): name_list += [actor.name_kana]
    #     if hasattr(actor, "full_name"): name_list += [actor.full_name]
    #     if hasattr(actor, "full_name_kana"): name_list += [actor.full_name_kana]
    #     response = self._cut_name_value(response, name_list, user_name).strip()
    #     # 文数でカット
    #     if sent_num > 0:
    #       response = self._cut_sentence_max(response, sent_num).strip()
    #       # print(f"***Response[cut_sentence]" + "*"*64 + f"\n{response}\n" + "*"*80)
    #     # 括弧で囲われていたらカット
    #     response = self._cut_brackets(response).strip()

    self.bef_response = response # 前回のレスポンスに保存

    return response

  # テンプレートタイプを取得
  def get_temp_type(self, template:str) -> int:
    if template == "Llama":
      temp_type = 0
    elif template == "Llama3":
      temp_type = 1
    elif template == "Auto-GPTQ":
      temp_type = 2
    elif template == "User-Assistant":
      temp_type = 3
    elif template == "Youri-chat":
      temp_type = 4
    elif template == "dolphin":
      temp_type = 5
    elif template == "Fugaku":
      temp_type = 6
    elif template == "Qwen2-VL":
      temp_type = 7
    elif template == "Idefics2":
      temp_type = 8
    else:
      temp_type = 99
    
    return temp_type

  # Llama3, Qwen2-VL, Idefics2用の文字列を作成
  def create_message(self, sys_prompt, message, actor_name, user_name, no_chat_key, tokenizer, processor=None, img_paths=False, vdo_paths=False, name_no_msg_flg=False):
    messages = [] 
    if sys_prompt and sys_prompt != "":
      messages.append({"role": "system", "content": sys_prompt})
 
    input_list = message.split("\n")
    set_input = ""
    role = "user" 
    for i, input in enumerate(input_list):
      if input.startswith(f"{actor_name}:") or \
         input.startswith(f"{user_name}:") or \
         input.startswith(no_chat_key):
          if role != "" and set_input != "":
            messages.append({"role": role, "content": set_input})
            set_input = ""
            role = ""

      if input.startswith(f"{actor_name}:"):
        role = "assistant"
      elif input.startswith(f"{user_name}:"):
        role = "user"
      elif input.startswith(no_chat_key):
        role = "user"

      # メッセージに名前を含めないときは名前が入っていたらカット
      if name_no_msg_flg:
        if input.startswith(f"{user_name}:"):
          input = input[len(f"{user_name}:"):].strip()
      
      set_input += f"\n{input}" if set_input != "" else input

      # 最終行の時は必ず追加
      if i == len(input_list) - 1:
        if vdo_paths:
          v_cnt_list=[]
          for v_path in vdo_paths:
            v_cnt_list.append({"type": "video", "video": v_path})
          v_cnt_list.append({"type": "text", "text": set_input})
          messages.append({"role": role, "content": v_cnt_list})
        elif img_paths:
          i_cnt_list=[]
          for i_path in img_paths:
            i_cnt_list.append({"type": "image", "image": i_path})
          i_cnt_list.append({"type": "text", "text": set_input})
          messages.append({"role": role, "content": i_cnt_list})
        else:
          messages.append({"role": role, "content": set_input})

    if processor:
      text = processor.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True # 推論時はTrue
      )
    else:
      text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True # 推論時はTrue
      )

    return text, messages

  # テンプレートタイプからプロンプトを作成
  def get_temp_prompt(self, temp_type:int, sys_prompt:str, message:str, images:str=None, videos:str=None, actor_name:str=None, user_name:str=None, no_chat_key:str=None, tokenizer=None, processor=None, name_no_msg_flg=False) -> str:
    prompt = ""
    messages= ""
    if temp_type == 0: # "Llama"
      prompt = rf"<s>[INST]\n<<SYS>>\n{sys_prompt}\n<</SYS>>\n{message}\n[/INST]"
    elif temp_type == 1: # "Llama3"
      prompt, messages = self.create_message(sys_prompt, message, actor_name, user_name, no_chat_key, tokenizer, None, None, name_no_msg_flg, replace_flg)
    elif temp_type == 2: # "Auto-GPTQ"
      prompt = rf"{sys_prompt} \n\n{message} ### 回答： "
    elif temp_type == 3: # "User-Assistant"
      prompt = rf"### USER: {sys_prompt} \n\n{message}\nASSISTANT: "
    elif temp_type == 4: # "Youri-chat"
      prompt = rf"設定: \n{sys_prompt}\n{message}"
    elif temp_type == 5: # dolphin
      prompt = rf"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant"
    elif temp_type == 6: # Fugaku
      prompt = rf"{sys_prompt}\n\n### 指示:\n{message}\n\n### 応答:\n"
    elif temp_type == 7: # Qwen2-VL
      prompt, messages = self.create_message(sys_prompt, message, actor_name, user_name, no_chat_key, tokenizer, processor, images, videos, name_no_msg_flg)
    elif temp_type == 8: # Idefics2
      prompt, messages = self.create_message(sys_prompt, message, actor_name, user_name, no_chat_key, tokenizer, processor, images, videos, name_no_msg_flg)
    else:
      prompt = f"{sys_prompt}\n{message}"
    return prompt, messages

  # チャット履歴を巻き戻す
  def history_rewind(self, rew_num=1, mode=0) -> bool: 
    history_dic_list = self.history_dic_list

    if len(history_dic_list) == 0:
      return False

    if len(history_dic_list) < rew_num:
      rew_num = len(history_dic_list)

    self.history_dic_list = history_dic_list[:-rew_num]

    return True

  # ★基本チャットレスポンス取得
  def _get_response(self, actor, message, sys_prompt=None, image_set=None, video_set=None, max_tokens=256, temperature=1.0, temp_type=0, direct_flg=False, no_log=False, user_name=None, no_chat_key=None, name_no_msg_flg=False):
    model, tokenizer = actor.model, actor.tokenizer
    
    if hasattr(actor, "processor"):
      processor = actor.processor

    if direct_flg:
      # メッセージを直接入力にする
      do_sample = False # ランダム性をなくす
      prompt = message
    else:
      do_sample = actor.do_sample
      if not sys_prompt:
        sys_prompt = ""

    # メッセージなしだとエラーになるので
    if message == "": message = " "

    # プロンプトを取得
    prompt, messages = self.get_temp_prompt(temp_type, sys_prompt, message, image_set[1], video_set[1], actor.chat_name, user_name, no_chat_key, tokenizer, processor, name_no_msg_flg)

    self.bef_prompt = prompt # 前回の入力プロンプトに保存
    response = ""
    
    if not no_log:
      logger.info("*** Full prompt ***\n" + prompt)

    try:
      with torch.no_grad():
        if model is not None:
          # model.device = model.to(device="cuda")
          # if images and len(images) > 0:
          if True:
            if (not image_set[0] and image_set[1]) or (not video_set[0] and video_set[1]):
              image_inputs, video_inputs = process_vision_info(messages)
            else:
              if not image_set or len(image_set) == 0:
                image_inputs = None
              else:
                image_inputs = image_set[0]
              if not video_set or len(video_set) == 0:
                video_inputs = None
              else:            
                video_inputs = video_set[0]

            if not image_inputs: image_inputs = None
            if not video_inputs: video_inputs = None

            token_ids = processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # token_ids = tokenizer.encode(text=[prompt], images=[image], return_tensors="pt", padding = True).to(model.device)
            # token_ids = processor(text=[prompt], images=images, return_tensors="pt", padding = True).to(model.device)
            output_ids = model.generate(
                **token_ids,
                max_new_tokens=max_tokens,
                # repetition_penalty = 1.05,
                repetition_penalty=actor.repetition_penalty, # 繰り返しを制限(1.0だと制限なし)
                # no_repeat_ngram_size=3,
                # top_k=50,  # 上位50トークンからサンプリング
                # top_p=0.95, # 上位95%の累積確率のトークンからサンプリング
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id         
                )
    
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(token_ids.input_ids, output_ids)
            ]
            response = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

    except Exception as e:
      print(f"Error: {str(e)}")
      print(f"Error: Failed to Get Response.")

    return response.strip()
  
# ◆TkinterでのGUIクラス
class Application(tk.Frame):

  # ステータスバー
  class StatusBar(tk.Frame):
    def __init__(self, master):
      super().__init__(master)
      self.label = tk.Label(self, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5)
      self.label.pack(fill=tk.X)

    def set_status(self, status_text):
      text = status_text.replace("\n","")
      self.label.config(text=text)

  # 初期設定
  def __init__(self, master, args, cfg):
    print(f"Started [{APP_NAME}]!")

    super().__init__()
    self.args = args
    self.cfg = cfg

    self.prop = Property(cfg)

    # self.actors = None
    self.load_img = None
    self.item_img = None
    self.actor_img = None # キャラの画像(キャンバス表示)
    self.chat_img = None # チャット画像(キャンバス表示)
    self.chat = None

    # ウィンドウ
    self.win_thought = None
    self.win_picture = None

    # デフォルト値を設定
    self._init_default()

    # 設定の画面にかかわる項目はここで設定
    self.redraw_msec = cfg.img_redraw_msec # メイン画面の再描画時間

    # ランダムシード値
    random.seed(set_random_seed(cfg, RANDOM_SEED))

    master.fonts = cfg.font_form # (cfg.font_form, 15)
    master.title(WINDOW_TITLE)

    form_w = cfg.master_size[0] if cfg.master_size[0] > 200 else 200
    form_h = cfg.master_size[1] if cfg.master_size[0] > 200 else 200

    screen_w = master.winfo_screenwidth()
    screen_h = master.winfo_screenheight()

    x = (screen_w - form_w) // 2
    y = (screen_h - form_h) // 2
    master.geometry(f"{form_w}x{form_h}+{x}+{y}")

    self.master = master

    # ステータスバーを追加
    self.status_bar = self.StatusBar(master)
    self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # -------------------------------------------------------------------------------
    # メニューバーの設定
    self.menubar = tk.Menu(root)
    self.menu_file = tk.Menu(self.menubar, tearoff=0)
    self.menu_edit = tk.Menu(self.menubar, tearoff=0)
    self.menu_view = tk.Menu(self.menubar, tearoff=0)

    self.menubar.add_cascade(label=MENU_FILE, menu=self.menu_file)
    self.menubar.add_cascade(label=MENU_EDIT, menu=self.menu_edit)
    self.menubar.add_cascade(label=MENU_VIEW, menu=self.menu_view)
    root.config(menu=self.menubar)

    # [ファイル]メニュー項目
    self.menu_file.add_command(label=MENU_CHAT_CLEAR, command=self.menu_clear_chat)
    self.menu_file.add_separator()
    self.menu_file.add_command(label=MENU_ACTOR_READ, command=self.menu_load_actor)
    self.menu_file.add_command(label=MENU_ACTOR_RELEASE, command=self.menu_release_actor)
    self.menu_file.add_separator()
    self.menu_file.add_command(label=MENU_CONFIG_RELOAD, command=self.menu_reload_config)
    self.menu_file.add_separator()
    self.menu_file.add_command(label=MENU_APP_END , command=self.menu_close)

    # [編集]メニュー項目
    self.menu_edit.add_command(label=MENU_RETAKE, command=self.menu_retake) # リテイク
    self.menu_edit.add_command(label=MENU_CHAT_BACK, command=self.menu_chat_back)
    self.menu_edit.add_separator()
    self.menu_edit.add_command(label=MENU_CHAT_EDIT, command=self.menu_chat_edit) # チャット編集

    # [表示]メニュー項目
    self.var_view_pict = tk.BooleanVar()
    self.menu_view.add_checkbutton(label=MENU_VIEW_PICTURE, variable=self.var_view_pict, command=self.menu_view_win_picture)

    # -----------------------------------------------------
    # メインパネル
    self.panel_main = tk.Frame(self.master, padx=5, pady=5)

    # 入力メッセージ用のパネル
    self.panel_01 = tk.Frame(self.panel_main, height=20)

    # PanedWindowを作成
    self.paned_window_1 = tk.PanedWindow(self.panel_main, orient=tk.HORIZONTAL)
    self.panel_02 = tk.Frame(self.paned_window_1, width=cfg.actor_win_size[0])
    self.panel_03 = tk.Frame(self.paned_window_1, width=cfg.master_size[0]-cfg.actor_win_size[0]-20)
    self.paned_window_2 = tk.PanedWindow(self.panel_02, orient=tk.VERTICAL)
    self.panel_02_1 = tk.Frame(self.paned_window_2)
    self.panel_02_2 = tk.Frame(self.paned_window_2)
    self.paned_window_3 = tk.PanedWindow(self.panel_03, orient=tk.VERTICAL)
    self.panel_03_1 = tk.Frame(self.paned_window_3)
    self.panel_03_2 = tk.Frame(self.paned_window_3, bg="gray40")

    self.paned_window_1.add(self.panel_02)
    self.paned_window_1.add(self.panel_03)
    self.paned_window_2.add(self.panel_02_1)
    self.paned_window_2.add(self.panel_02_2)
    self.paned_window_3.add(self.panel_03_1)

    self.panel_main.pack(fill=tk.BOTH, expand=True)
    self.panel_01.pack(side=tk.TOP, fill=tk.X, pady=(5, 5))
    self.paned_window_1.pack(fill=tk.BOTH, expand=True)
    self.paned_window_2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    self.paned_window_3.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # 入力メッセージエントリーボックスを作成
    self.etr_msg = tk.Entry(self.panel_01)

    # 送信ボタンを作成
    self.btn_send = tk.Button(self.panel_01, width=10, text=BUTTON_SEND, command=self.on_btn_send_click)

    # キャラクターメイン画像用キャンパスを作成
    self.cnv_actor = tk.Canvas(self.panel_02_1, width=cfg.actor_win_size[0], height=cfg.actor_win_size[1], bg="gray20")

    # キャラクタープロフィール用テキストを作成
    self.txt_prof = tk.Text(self.panel_02_2, width=0, fg="white", bg="gray20")

    # チャット内容用テキストボックスを作成
    self.scrollbar = tk.Scrollbar(self.panel_03_1)  # Scrollbarを作成
    self.txt_chat = tk.Text(self.panel_03_1, height = 150,font=cfg.font_chat, fg="black", bg=self.cfg.color_txt_chat, undo=True, highlightthickness=1, yscrollcommand=self.scrollbar.set)
    self.txt_chat.bind('<Control-Shift-Key-Z>', self.redo) # Redoをバインド

    # チャット内容用キャンパスを作成
    # self.cnv_chat_f = ttk.Frame(self.panel_03_2, bg="gray40", highlightthickness=1)
    self.cnv_chat = tk.Canvas(self.panel_03_2, bg="gray40", highlightthickness=1)

    self.etr_msg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.btn_send.pack(side=tk.RIGHT)
    self.cnv_actor.pack(fill=tk.BOTH, expand=True)
    self.txt_prof.pack(fill=tk.BOTH, padx=1, expand=True)
    self.txt_prof.config(state=tk.DISABLED)  # テキストボックスを非活性にする
    self.txt_prof.config(font=cfg.font_prof)

    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.txt_chat.pack(fill=tk.BOTH, expand=True)
    self.cnv_chat.pack(fill=tk.BOTH, expand=True)

    self.txt_chat.config(state=tk.DISABLED)  # テキストボックスを非活性にする

    # ScrollbarとTextコントロールを連携
    self.scrollbar.config(command=self.txt_chat.yview)

    # チャットクラス
    self.chat = Chat()
    self.chat.owner = self
    # self.chat.logger = self.logger
    self.chat.print_log = print_log
    # self.is_chat_now = False
    self.etr_msg.focus_set()

    # 変換実行中フラグ
    self.conv_now_flg = False

    # チャット書き込み中フラグ
    self.is_write_chat = False

    # キーEnterイベントを設定
    self.etr_msg.bind("<Return>", self.press_key)

    # ウィンドウの閉じるボタンが押されたときの処理
    self.master.protocol("WM_DELETE_WINDOW", self.menu_close)

    # ふきだしチャット
    self.bub_chat = None
    # キャラ画像ウィンドウ
    self.win_actor_img = None
    # 思考ウィンドウ
    self.win_thought = None
    # 追加指示ウィンドウ
    self.win_instruct = None
    # 入力画像ウィンドウ
    self.win_picture = None

    # キャラクターをロード(デフォルト)
    self.main_img = None # メインイメージ
    self.main_img_gst = None # メインイメージ(ゲスト)
    self.actors = [None]

    self.event_load_num = 0 # イベントモードの初期表示行
    self.send_message_list = [] # 送信する行
    self.bef_message = ""
    self.last_message = "" # 保存(この値が変わっていたら前回の問い合わせはキャンセルされる)
    self.last_send_message = "" # 最後に送信したメッセージ
    self.receive_msg_dic = {} # 戻ってきた辞書形式のメッセージ

    self.actor_path = cfg.def_actor
    if args.model_path:
      self.actor_path = args.model_path
    self.master.after(0, lambda: self.start_load_actor(self.actor_path))

    # チャット表示モード(0:特殊文字非表示, 1:特殊文字表示)
    self.view_chat = 0

    # テキストボックスにタグ付け
    self.txt_chat.tag_configure(self.cfg.user_name, foreground=self.cfg.chat_color)
    if hasattr(self.cfg, "other_names") and hasattr(self.cfg, "other_colors"):
      for other_name, other_color in zip(cfg.other_names, cfg.other_colors):
        self.txt_chat.tag_configure(other_name, foreground=other_color)

    # キャラクター画像表示(ループ)
    self.master.after(0, lambda: self.actor_img_update())

    # ドラッグ＆ドロップの設定
    self.master.drop_target_register(DND_FILES)
    self.master.dnd_bind('<<Drop>>', self.on_drop)

  # 初期化中のデフォルト値
  def _init_default(self):
    self.command_view_flg = False
    self.graph_view_flg = False
    self.chat_edit_flg = False
    self.load_img = None
    self.chat_thread = None
    self.load_thread = None

    self.chat_stop_event = threading.Event()
    self.load_stop_event = threading.Event()
    self.chat_stop_event.clear()

    # self.load_img_tags = None
    # self.item_img = None
    # self.item_img_tags = None
    # self.train_thread = None
    # self.think_stop_event = threading.Event()
    # self.train_stop_event = None
    # self.test_stop_event = None
    # self.think_stop_event.clear()
    # self.load_stop_event.clear()

  # 設定で自動的に画面を開く
  def open_window(self, actor):
    # 入力画像ウィンドウを開く
    if actor:
      if actor.def_picture:
        if not self.var_view_pict.get():
          self.var_view_pict.set(True)
          self.win_picture = PictureWindow(self)

  # 効果文字列をリストに変換
  def conv_effect_list(self, text):
    eff_texts = []
    for value in EFFECT_STRING:
      pattern = re.compile(fr"\[{value}:.*?\]")
      matches = re.findall(pattern, text)
      eff_texts.extend(matches)
      text = re.sub(pattern, '', text).strip()
    return eff_texts, text

  # フォルダのドラッグドロップ
  def on_drop(self, event):
    dir_path = event.data  # ドロップされたファイルのパスを取得
    if os.path.isdir(dir_path): 
      self.menu_load_actor(dir_path)
    else:
      print("Warning: Input actor Folder.")

  # メッセージを表示(ステータスバーorメッセージボックス)
  def message_show(self, message, title=None, timer_flg=True):
    if self.cfg.status_bar_flg:
      self.status_bar_msg(message, timer_flg)
    else:
      messagebox.showinfo(MSG_TTL_INFO, message)

  # ステータスバーにメッセージを表示(設定時間後に非表示)
  def status_bar_msg(self, message, Timer_flg=True):
    self.status_bar.set_status(message)
    if Timer_flg:
      self.master.after(self.cfg.status_bar_msg_msec, self.status_bar_clear)

  # ステータスバーのメッセージを削除
  def status_bar_clear(self):
    self.status_bar.set_status("")

  # チャットテキストのやり直し
  def redo(self, event=None):
    try:
      self.txt_chat.edit_redo()
    except:
      pass # やり直しできないときはエラー

  # メニューのラベルを変更
  def change_menu_item_name(self, menu_item, old_name, new_name):
    try:
      index = menu_item.index(old_name)  # ラベル名を使用してインデックスを取得
      menu_item.entryconfig(index, label=new_name)  # インデックスを使用して名前を変更
    except ValueError:
      print(f"Error: Label [{old_name}] was not found.")

  # 履歴の削除を確認
  def confirm_memory_delete(self, message) -> bool:
    result = messagebox.askyesno(MENU_CHAT_CLEAR, message + f"\n※履歴もすべて削除されます。")
    if not result:
      return False
    return True

  # [メニュー]チャットのクリア
  def menu_clear_chat(self):
    if len(self.chat.view_history) == 0:
      return

    result = messagebox.askyesno(MENU_CHAT_CLEAR, f"チャット内容をクリアしますか？\n※履歴もすべて削除されます。")
    if not result:
      return

    self.clear_chat()

  # [メニュー]キャラクター読込
  def menu_load_actor(self, dir_path=False):
    actor = self.actors[0]

    if dir_path:
      actor_dir_path = dir_path
    else:
      open_dir = None
      if os.path.isdir(self.actor_path):
        open_dir = self.actor_path

      actor_dir_path = filedialog.askdirectory(initialdir=open_dir)

    req_files = [self.cfg.actor_data] # 必須ファイル
    if actor_dir_path:
      for file in req_files:
        if not os.path.isfile(os.path.join(actor_dir_path, file)):
          messagebox.showerror(MSG_TTL_ERR, "モデルファイル({file})が見つかりませんでした。")
          self.actor_menu_enable(False)
          return

      # フォームの読み込みを中止
      self.load_stop_event.set()

      if actor:
        # パラメーターを保存
        actor.save_parameter()

        # モデル解放
        actor.release_model()

      # 新しいモデルを読み込み
      ret = self.load_actor(actor_dir_path, 0)

      actor = self.actors[0]
      if ret:
        self.actor_path = actor_dir_path

        if actor.model_enable:
          if self.cfg.thread_model_load:
            # キャラクタロード時にモデルを別スレッドでロード
            self.message_show("モデル設定ファイルを読み込みました。\nモデルは別スレッドで読み込み中です。", MSG_TTL_INFO)
          else:
            self.message_show("モデル設定ファイルを読み込みました。", MSG_TTL_INFO)
        
        self.actor_menu_enable(True)

      if not actor:
        self.actor_menu_enable(False)
        messagebox.showerror(MSG_TTL_ERR, "モデルの読み込みに失敗しました。")

  # [メニュー]モデル解放
  def menu_release_actor(self):
    if not self.actors[0]:
      return

    # モデル読み込みの終了を待つ
    if self.load_thread and self.load_thread.is_alive():
      print("Model-load-Thread is still Running. Forcefully Terminating...")
      # スレッドに停止の合図を送る
      self.load_stop_event.set()
      self.load_thread.join()

    # モデルメインループを止める
    if self.actors[0].main_thread and self.actors[0].main_thread.is_alive():
      print("actor-Main-Thread is still Running. Forcefully Terminating...")
      # スレッドに停止の合図を送る
      self.actors[0].main_stop_event.set()
      self.actors[0].main_thread.join()

    # チャットスレッドの終了を待つ
    if self.chat_thread and self.chat_thread.is_alive():
      print("Chat-Thread is still Running. Forcefully Terminating...")
      # スレッドに停止の合図を送る
      self.chat_stop_event.set()
      self.chat_thread.join()

    self.actors[0].release_model()
    # del self.actors[0]
    # self.actors.append(None)
    self.actors[0] = None
    gc.collect()

    self.actor_img = None
    self.main_img = None
    self.txt_prof.config(state=tk.NORMAL)
    self.txt_prof.delete("1.0", "end")
    self.txt_prof.config(state=tk.DISABLED)

    self.master.title(f"{WINDOW_TITLE}")

    self.actor_menu_enable(False)
    messagebox.showinfo(MSG_TTL_INFO, "モデルを開放しました。")

  # [メニュー]モデル情報再読込
  def menu_reload_actor_info(self):
    actor_path = self.actor_path
    if not os.path.isdir(actor_path):
      messagebox.showerror(MSG_TTL_ERR, "モデルが読み込まれていません。")
      return

    req_files = [self.cfg.actor_data] # 必須ファイル
    for file in req_files:
      if not os.path.isfile(os.path.join(actor_path, file)):
        messagebox.showerror(MSG_TTL_ERR, "モデルファイル({file})が見つかりませんでした。")
        self.actor_menu_enable(False)
        return

      # self.actors[0].save_parameter()

      # キャラ情報のみ更新
      ret = self.actors[0].load_config(no_param_flg=True)

      self.open_window(self.actors[0])

      if ret:
        # self.actors[0].load_parameter()

        # 高速モデルの有無
        state = tk.NORMAL if self.actors[0].conv_ct2 else tk.DISABLED
        # self.menu_tool.entryconfig(MENU_MODEL_CONVERT, state=state)

        # キャラクタロード時にモデルを別スレッドでロード
        messagebox.showinfo(MSG_TTL_INFO, "モデル情報のみを再読み込みしました。\nモデルは再ロードされていません。")
      else:
        messagebox.showinfo(MSG_TTL_ERR, "モデル情報の再読み込みに失敗しました。")
 
  # [メニュー]設定再読み込み
  def menu_reload_config(self):
    self.cfg = Config(CONFIG_FILE)
    self.redraw_msec = self.cfg.img_redraw_msec # メイン画面の再描画時間
    messagebox.showinfo(MSG_TTL_INFO, f"設定ファイル({CONFIG_FILE})を再読み込みしました。")

  # [メニュー]ファイル：閉じる
  def menu_close(self):
    # モデルを解放
    for c in self.actors:
      if not c:
        try:
          c.release_model()
        except Exception:
          pass
    del self.actors

    exit()

  # [メニュー]編集：リテイク
  def menu_retake(self):
    if self.chat_edit_flg:
      # チャットテキスト編集後のとき
      text = "チャットテキスト編集を確定し、\nチャットをリテイクしますか？"

      result = messagebox.askyesno(MSG_TTL_CONFIR, text)
      if not result:
        return

      # 編集確定
      self.chat_edit_end()
    
    history_list = self.chat.history_dic_list
    if len(history_list) == 0:
      return

    n = 2
    if len(history_list) < n:
      n = 1
    
    if n > 1:
      name = history_list[-n]["name"]
      message = history_list[-n]["text"]
      eff_list = history_list[-n]["effects"]
    else:
      name = None
      message = ""
      eff_list = None
    
    if eff_list and len(eff_list) > 0:
      message += " " + " ".join(eff_list)

    # [メニュー]編集：チャットを一つ戻る
    self.chat_back(num=n)

    self.chat_stop_event.clear()

    # 書き込むだけのとき、書き込んで終了
    if self.is_write_only_chat(self.cfg, message):
      return

    self.chat_thread = threading.Thread(target=self.send_message_chat, args=([name, message, self.actors, 1, self.chat_stop_event, True, False, False]), daemon=True)
    self.chat_thread.start()

  # [メニュー]編集：チャットを一つ戻る
  def menu_chat_back(self):
    if self.chat_edit_flg:
      # チャットテキスト編集後のとき
      text = "チャットテキスト編集を確定し、\nチャットを一つ戻りますか？"

      result = messagebox.askyesno(MSG_TTL_CONFIR, text)
      if not result:
        return

      # 編集確定
      self.chat_edit_end()

    ret = self.chat_back()

    if not ret:
      messagebox.showwarning(MSG_TTL_INFO, "チャット履歴が存在しません。")
    
  # [メニュー]チャットテキスト編集
  def menu_chat_edit(self):
    if self.chat_edit_flg:
      # チャットテキスト編集後のとき
      result = messagebox.askyesno(MSG_TTL_CONFIR, f"チャットテキスト編集を確定しますか？")
      if result:
        self.chat_edit_end()
      return

    # 確認
    result = messagebox.askyesno(MSG_TTL_CONFIR, f"チャットテキスト編集しますか？\n※チャット開始時に編集内容が確定されます。")
    if not result:
      return

    self.chat_edit_flg = True

    # チャット表示
    self.view_chat = 1
    self.update_chat()

    self.txt_chat.config(state=tk.NORMAL, bg="white")

    # [チャット履歴]非活性
    # self.menu_view_chat_enable(False)

    # ラベル変更
    self.change_menu_item_name(self.menu_edit, MENU_CHAT_EDIT, MENU_CHAT_EDIT_CONF)

  # [メニュー]ウィンドウ：入力画像
  def menu_view_win_picture(self):
    if self.var_view_pict.get():
      # 新しいウィンドウとしてDndWindowを開く
      self.win_picture = PictureWindow(self)
    else:
      self.win_picture.on_closing()

  # [ボタン] 送信：クリックイベント
  def on_btn_send_click(self):

    if self.chat_edit_flg:
      # チャットテキスト編集後のとき
      result = messagebox.askyesno(MSG_TTL_CONFIR, f"チャットテキスト編集を確定し、\nチャットを開始しますか？")
      if not result:
        return
      self.chat_edit_end()

    # メッセージを取得して一時的に非活性
    message = self.etr_msg.get().strip()
    # self.send_control_activ(False)

    # 書き込むだけのとき、書き込んで終了
    if self.is_write_only_chat(self.cfg, message):
      return

    # チャットキャンセル
    if message == "": # チャットEXのときはメッセージが空白のときのみキャンセル可能
      if self.chat_thread and self.chat_thread.is_alive():
        result = messagebox.askyesno(MSG_TTL_CONFIR, "チャットを中断しますか？")
        if result:  
          # 中断されるまでボタンを操作不可にする
          self.send_menu_enabled(False)
          self.etr_msg.config(state=tk.DISABLED)
          self.btn_send.config(state=tk.DISABLED)

          # チャットを中断
          self.chat_stop_event.set()
        return

    # メッセージを入力エントリーに戻す用
    self.bef_message = message

    # 保存(この値が変わっていたら前回の問い合わせはキャンセルされる)
    self.last_message = f"{self.cfg.user_name}: {message}"

    ex_flg = True
    target = self.send_message_chat
    args = [None, message, self.actors, 1, self.chat_stop_event, False, False, False, False, False, ex_flg]

    # 入力メッセージのエントリーの内容を削除
    self.etr_msg.delete(0, "end")
    
    self.chat_stop_event.clear()
    self.chat_thread = threading.Thread(target=target, args=(args), daemon=True)
    self.chat_thread.start()

  # キープレスイベント(Enter)
  def press_key(self, _):
    # if not self.is_chat_now:
    if self.btn_send['state'] == tk.NORMAL:
      # ボタンが活性のときのみ送信可
      self.on_btn_send_click()

  # チャット欄に書き込むだけのときは書き込んで「True」を返す
  def is_write_only_chat(self, cfg, message) -> bool:
    if hasattr(cfg, "no_chat_key"):
      if cfg.no_chat_key:
        if message.startswith(self.cfg.no_chat_key):
          # 直接入力文字が先頭に入っていたら、そのまま記入(履歴には保存)
          eff_texts = None
          # 画像だけは取得
          if self.var_img_record.get():
            # 入力画像とパスを入力画像ウィンドウから取得
            img_max = self.cfg.input_image_max if hasattr(self.cfg, "input_image_max") else -1
            _, input_img_paths = self.get_input_image(img_max, True)
            
            # 画像パスを追加
            eff_texts = self.get_image_eff_list(input_img_paths)

          # 前の画像パスを削除
          if "<Image:" in message:
            img_path_ptn = re.compile(r"\<Image:(.*?)\>")
            message = img_path_ptn.sub("", message)
          img_path_ptn = re.compile(r"\[Image:(.*?)\]")
          message = img_path_ptn.sub("", message)
          # 履歴に追加(+チャット欄にも追加)
          self.add_txt_chat(None, message, eff_texts=eff_texts, chat_update=True)
          self.etr_msg.delete(0, "end")
          self.etr_msg.config(state=tk.NORMAL) # 入力を活性

          return True

    return False

  # チャットテキスト編集を終了
  def chat_edit_end(self, label_change_flg=True):
    # 編集後の全テキストデータ(リスト)
    txt_chat_line= self.txt_chat.get("1.0", "end-1c").rstrip().split("\n")

    new_chat_line_list = []
    new_chat_line_list_view = []
    new_chat_line_list_simple = []

    # 新しい履歴用
    new_history_dic_list = [] 
    new_history_dic = {}
    text_list = []
    eff_list = []
    res_list = []
    name_list = [self.cfg.user_name]
    for actor in self.actors:
      name_list.append(actor.chat_name)
    name_list.extend(self.cfg.other_names)

    # 重複を省く
    name_list = list(set(name_list))

    if True:
      bef_name = None
      for text_chat in txt_chat_line:
        text = text_chat.rstrip() # 末尾のみ空白を削除
        tmp_text = text

        # 新しい履歴への追加
        tmp_name = None
        add_flg = False
        if tmp_text.startswith(self.cfg.no_chat_key) or \
           tmp_text.startswith(self.cfg.chat_split_key):
          tmp_name = ""
          add_flg = True
          
        if not add_flg:
          for name in name_list:  
            if text.startswith(f"{name}:"):
              tmp_name = name
              tmp_text = text[len(f"{name}:"):].strip()
              add_flg = True            
              break

        if add_flg:
          if bef_name or bef_name == "":            
            new_history_dic["name"] = bef_name if bef_name != "" else None
            new_history_dic["text"] = "\n".join(text_list)
            new_history_dic["effects"] = eff_list
            new_history_dic_list.append(new_history_dic)
            new_history_dic={}
            text_list, eff_list, res_list = [], [], []
        if tmp_name:
          bef_name = tmp_name
        elif tmp_name == "":
          bef_name = ""

        # match_res = re.findall(pattern_t, text)        
        # text = re.sub(pattern_t, '', text).strip() # 応答時間を削除

        text_simple = text
        text_view = text
        
        # res_list.extend(match_res) # 新しい履歴は応答時間を取得
        # tmp_text = re.sub(pattern_t, '', tmp_text).strip() # 応答時間を削除

        # 正規表現を使用して指定の範囲を空文字列に置換
        for value in EFFECT_STRING:
          pattern = re.compile(fr"\[{value}:.*?\]")
          # シンプル表示はすべての効果文字を取り除く
          if value == "Word" \
              or value == "Datetime" \
              or value == "Image":
            # 画像は[]の文字が入っている可能性があるので、
            if "<Image:" in text_view:
              pattern_2 = re.compile(fr"\<Image:.*?\>")
              text_simple = re.sub(pattern_2, '', text_simple).strip()
              text_view = re.sub(pattern_2, '', text_view).strip()
            
            # ビュー表示はいくつかの効果文字を取り除く
            text_simple = re.sub(pattern, '', text_simple).strip()
            text_view = re.sub(pattern, '', text_view).strip()

          # 新しい履歴
          if value == "Image" and "<Image:" in tmp_text:
            pattern_2 = re.compile(fr"\<Image:.*?\>")
            matches_2 = re.findall(pattern_2, tmp_text)
            eff_list.extend(matches_2)
            tmp_text = re.sub(pattern_2, '', tmp_text).strip() # 削除

          matches = re.findall(pattern, tmp_text)
          eff_list.extend(matches)
          tmp_text = re.sub(pattern, '', tmp_text).strip() # 削除

        new_chat_line_list.append(text)
        new_chat_line_list_view.append(text_view)
        new_chat_line_list_simple.append(text_simple)

        # 新しい履歴(名前なし)
        text_list.append(tmp_text)

      self.chat.view_history = "\n".join(new_chat_line_list)
      self.chat.view_history_view = "\n".join(new_chat_line_list_view)
      self.chat.view_history_simple = "\n".join(new_chat_line_list_simple)
      self.chat.view_history_time = "\n".join(new_chat_line_list) # 処理時間はテキストデータになる

      # 新しい履歴(最後の一行)
      new_history_dic["name"] = bef_name if bef_name != "" else None
      new_history_dic["text"] = "\n".join(text_list)
      new_history_dic["effects"] = eff_list
      # if len(res_list) == 0:
      #   new_history_dic["response"] = None
      # else:
      #   new_history_dic["response"] = res_list[-1]
      new_history_dic_list.append(new_history_dic)
      self.chat.history_dic_list = new_history_dic_list

    self.txt_chat.config(bg=self.cfg.color_txt_chat)
    self.view_chat = 0 # 前の状態に戻す
    self.update_chat() # チャット内容の更新

    # ラベル[チャットテキスト編集]を戻す
    if label_change_flg:
      self.change_menu_item_name(self.menu_edit, MENU_CHAT_EDIT_CONF, MENU_CHAT_EDIT)

    self.chat_edit_flg = False

  # チャットを戻す
  def chat_back(self, num=None) -> bool:
    mode = 1 # self.var_mode.get()
    if not num:
      n = 2
    else:
      n = num

    ret = self.chat.history_rewind(rew_num=n, mode=mode)

    if ret:
      self.update_chat()

    return ret

  # キャラクターの画像を更新(ループ処理)
  def actor_img_update(self):
    try:
      canvas = self.cnv_actor
      if self.win_actor_img:
        self.cnv_actor.delete("all")
        # canvas = self.win_actor_img.canvas
      else:
        # if self.actors:
        if not self.main_img:
          canvas.delete("all")
        else:
          # リサイズ後の画像を取得　※何も使用していない
          self.actor_img = self.draw_canv_img(self.main_img, canvas)

    except Exception as e:
      self.cnv_actor.delete("all")
      # print(f"Error: {str(e)}")

    self.master.after(self.redraw_msec, lambda: self.actor_img_update())

  # チャットのクリア
  def clear_chat(self):
    self.chat.view_history = ""
    self.chat.view_history_view = ""
    self.chat.view_history_simple = ""
    self.chat.view_history_time = ""
    
    self.chat.history_dic_list.clear()
    self.update_chat()

    # メモリー出力フラグもOFFに
    self.mem_output_flg = False
    self.mem_out_row = 0
    
    # 送信予定リストもクリア
    self.send_message_list.clear()

  # 画像のクリア
  def menu_clear_image(self):
    del self.load_img
    self.load_img = None
    self.load_img_tags = None
    self.chat_img = self.draw_canv_img(self.load_img, self.cnv_chat)
    self.cnv_chat.delete("all")
    self.master.update()

  # ★[チャット] メッセージを送信して返信を得る
  def send_message_chat(self, msg_name, message:str, actor_list:list, res_count=1, stop_event:threading.Event=None, retake=False, start_flg=False, event_flg=False, think_flg=False, ex_flg=False, inpre_flg=False):
    if not think_flg:
      if self.command_view_flg:
        # コマンドプロンプト画面から戻る
        self.change_command_view(False)
        self.return_cnv_chat()

      # 返信待ち中に各項目を非活性にする
      self.send_menu_enabled(False)

      # 先頭にユーザ名が入っていたら削除
      if message.startswith(f"{self.cfg.user_name}:"):
        message = message[len(f"{self.cfg.user_name}:"):].strip()

    try:
      if len(actor_list) > 1:
        random.shuffle(actor_list)

      for _ in range(res_count):
        for i, actor in enumerate(actor_list):

          # ★チャットの内部処理：エラーでも中断でも後処理は必要なので別スレッドにする
          if not think_flg: actor.is_chat_now = True
          result = self.chat_proc(self.cfg, actor, msg_name, message, None, stop_event, retake, start_flg, event_flg, think_flg, ex_flg=ex_flg)
          if not think_flg: actor.is_chat_now = False

          message = "" 
                   
          # メッセージを履歴に書き込み後に途中終了されていたら履歴を戻す
          if result is None:
            self.chat_restoration(rew_num=i+1)

          actor.last_action_time = datetime.datetime.now()
          if not cfg.multi_chat_flg or think_flg or event_flg:
            # マルチチャットフラグOFF or 思考 or イベントのときは一人目で終了
            break

    except Exception as e:
      print("Failed to Get Chat Response.")
      print(f"Error: {str(e)}")
      traceback.print_exc()

    # [チャット履歴]活性
    # self.menu_view_chat_enable(True)

    # [送信]ボタン活性　
    self.send_menu_enabled(True)

  # 画像のサイズを変更(最大値に合わせる)
  def resize_input_image(self, input_img:Image, img_max:int) -> Image:
    if img_max == -1:
      return input_img
    
    if input_img.width > img_max or input_img.height > img_max:
      # 縦横比を保ったリサイズ
      ratio = input_img.width / input_img.height
      new_width, new_height = input_img.width, input_img.height
      if input_img.width > input_img.height:
        new_width = img_max
        new_height = new_width / ratio
      else:
        new_height = img_max
        new_width = new_height * ratio
      
      new_img = input_img.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
    else:
      new_img = input_img
    return new_img

  # チャット時のモデルの読み込み待機
  def proc_load_model(self, actor, stop_event):
    # モデル読み込み中の時は待機
    while actor.model_loading_flg:
      time.sleep(0.1)

    # モデル有効キャラ
    if actor.model_enable:
      if not actor.model:
        # モデルがまだ読み込まれていなかったら読込
        self.status_bar_msg("モデルを読み込んでいます...")
        actor.load_model(stop_event=stop_event)
        # 読み込みに失敗したら終了
    if not actor.model:
      print_log("Error: Failed to Load Model.", log_type=1)
      return False
        
    return True

  # ★[チャット] チャット処理内部
  def chat_proc(self, cfg:Config, actor:actor, msg_name, message, set_sys_prompt=None, stop_event:threading.Event=None,  retake=False, start_flg=False, event_flg=False, think_flg=False, no_view=False, ex_flg=False) -> Optional[str]:
    # ユ－ザー名
    user_name = cfg.user_name

    if msg_name:
      user_name = msg_name
    else:
      msg_name = actor.chat_name if message.startswith(f"{actor.chat_name}:") else None
      if not msg_name:
        msg_name = actor.name if message.startswith(f"{actor.name}:") else None
      if not msg_name:
        # 先頭にサブ名が入っていたら、その名前で発言
        msg_name = next((name for name in cfg.other_names if message.startswith(f"{name}:")), None)
      if msg_name:
        user_name = msg_name
        message = message[len(user_name) + 1:].strip()

    # キャラのシステムプロンプトを取得
    actor_name = actor.chat_name

    # 最大文数
    if actor.max_sentences:
      sent_num = actor.max_sentences
    else:
      # キャラごとの最大文数が設定されていなかったら、共通のものを使う
      sent_num = cfg.max_sentences if not cfg.chat_hist_flg else cfg.max_sentences_hist

    # 会話間隔
    if start_flg:
      new_line = 0
    else:
      new_line = cfg.new_line_num if hasattr(cfg, "new_line_num") else 0

    key_word = None
    # img_path = None
    eff_texts = []

    # Llama3の名前置換フラグ
    # replace_flg = cfg.header_id_replace_flg if hasattr(cfg, "header_id_replace_flg") else False
    name_no_msg_flg = self.prop.get_property(actor, "name_no_msg_flg", False) 
    # header_id_replace_flg = self.prop.get_property(actor, "header_id_replace_flg", False) 
    # full_response_flg = self.prop.get_property(actor, "full_response_flg", False) 
    
    # 仮メッセージ
    tmp_message = message
    no_print = False


    if tmp_message:
      # 特殊効果文字列を抜き出す
      for value in EFFECT_STRING:
        pattern = re.compile(rf"\[{value}:.*?\]")
        if value == "Image" and "<Image:" in tmp_message:
          pattern = re.compile(rf"\<Image:.*?\>")

        matches = re.findall(pattern, tmp_message)
        eff_texts.extend(matches)
        tmp_message = re.sub(pattern, '', tmp_message).strip()

        # 画像パスを取得
        # if value == "Image" and len(matches) > 0:
        #   if "<Image:" in matches[0]:
        #     img_name = matches[0].split("<Image:")[1].split(">")[0].strip()
        #   else:
        #     img_name = matches[0].split("[Image:")[1].split("]")[0].strip()

    my_msg = tmp_message
    if not retake:
      # リテイクのときはそのまま
      if not tmp_message is None:
        if tmp_message == "":
          # my_msg = ""
          if len(eff_texts) == 0:
            my_msg = f"" # メッセージなし
        else:
          # my_msg = f"{user_name}: {tmp_message}"
          my_msg = tmp_message

    # Noneのときはメッセージなしと同じ
    if not my_msg or my_msg == "":
      my_msg = ""
      my_msg_user = ""
    else:
      my_msg_user = f"{user_name}: {my_msg.strip()}"

    # 日付日時を追加
    if hasattr(cfg, "chat_datetime_flg") and cfg.chat_datetime_flg:
      # 既にあったら削除してから
      # eff_texts = [datetime_ptn.sub("", item) for item in eff_texts]
      eff_texts = [item for item in eff_texts if not datetime_ptn.search(item)] # 時刻
      eff_texts = [item for item in eff_texts if not img_path_ptn.search(item)] # 画像
      eff_texts.append(get_datetime_chat())

    # 途中終了
    if stop_event.is_set(): return None

    # 入力削除
    # if not discussion_flg:
    #   self.etr_msg.config(state=tk.NORMAL)
    #   self.etr_msg.delete(0, "end")
    #   self.etr_msg.config(state=tk.DISABLED)
    # if hasattr(cfg, "no_top_msg_flg") and cfg.no_top_msg_flg:
    #   top_msg =""
    # else:
    #   top_msg = "＜会話＞"

    top_msg =""
    if cfg.chat_hist_flg:
      # キャラに渡すチャット履歴数を制限
      # if not self.chat.view_history_simple or len(self.chat.view_history_simple) == 0:
      #   talk_history_list = []
      # else:

      # こっちを取得しないと履歴が途中までしか取得できない可能性あり
      talk_history_list = [f"{dic['name']}: {dic['text']}" if dic["name"] else f"{dic['text']}" for dic in self.chat.history_dic_list if dic["name"] or (dic["text"] and not dic["text"].startswith(cfg.chat_split_key))]

      len_hist = len(talk_history_list)
      if len_hist > actor.chat_remenb_num:
        talk_history_list = talk_history_list[len_hist-actor.chat_remenb_num:]

      # [Datetime:]部分を削除
      # 会話履歴に対して正規表現にマッチする部分を空文字列に置換
      talk_history_list = [datetime_ptn.sub("", item) for item in talk_history_list]

      # [Image:]部分を削除
      talk_history_list = [img_path_ptn.sub("", item) for item in talk_history_list]

      send_message = top_msg

      if len(talk_history_list) != 0:
        if send_message != "": send_message += "\n"
        send_message += SEPARATED_VALUE.join(talk_history_list)
      if my_msg_user != "":
        if send_message != "": send_message += "\n"
        send_message += my_msg_user + "\n"

    # [Datetime:][Image:]はメッセージに入れない。
    eff_texts_s = [item for item in eff_texts if not datetime_ptn.match(item) and not img_path_ptn.match(item)]
    send_message += "" if len(eff_texts_s) == 0 else "\n" + " ".join(eff_texts_s)

    # 入力画像とパスを入力画像ウィンドウから取得
    input_images, input_img_paths = self.get_input_image(actor.input_image_max, not think_flg)
    input_img_set = [input_images, input_img_paths]
 
    # 動画
    input_vdo_set = [None, None]

    # 画像パスを追加
    if len(input_img_paths) > 0 and not event_flg:
      eff_texts.extend(self.get_image_eff_list(input_img_paths))

    if not think_flg and not event_flg:
      chk_message = f"{user_name}: {my_msg}"
      self.last_message = chk_message
    else:
      chk_message = None

    if my_msg.strip() != "":
      self.add_txt_chat(user_name, my_msg, eff_texts=eff_texts, chat_update=True, new_line_num=new_line, font_tag=user_name)
      logger.info(f"*** Message ***\n" + my_msg_user)
    
    template = "Llama"
    if hasattr(actor, "template"):
      template = actor.template

    # テンプレートタイプ
    temp_type = self.chat.get_temp_type(template)

    # システムプロンプト
    if set_sys_prompt:
      sys_prompt = set_sys_prompt
    else:
      sys_prompt = actor.base_sys_prompt

    send_message = send_message.strip()

    if not template in ["Llama3", "Qwen2-VL", "Idefics2"]:
      send_message += f"\n{actor_name}: " # メッセージの最後にキャラ名を追加

    if not think_flg:
      # 思考中でないとき
      # 送信メッセージ(全文)を表示
      send_message_print = '\n'.join(['* ' + s for s in send_message.split("\n")])
      # print_log(f"*** Full Send Message " + "*"*78 + f"\n{send_message_print}\n" + "*"*100,  log_new_line=True, print_no_type=True)
      print(f"*** Full Send Message " + "*"*78 + f"\n{send_message_print}\n" + "*"*100)
      logger.info(f"*** Full Send Message ***\n" + send_message)
      print(my_msg_user)

    s_time = time.time() # 推論開始時間

    try_num = 0
    try_max = cfg.word_try_max if hasattr(cfg, "word_try_max") else 9999

    # モデルが読み込まれていなかったらこの時点で待機
    if not self.proc_load_model(actor, stop_event):
      return None

    while True:
      try_num += 1
      # 途中終了
      if stop_event.is_set(): return None
      # if not think_flg and not event_flg:
      if self.check_msg_change(chk_message): return ""
      
      if key_word:
        print_log(f"[Response No.{try_num}] ... ", no_new_line=True, print_no_type=True)

      sys_prompt_t = sys_prompt
      # if not think_flg:
      #   # チャット中のキャラの思考を取得
      #   if actor.chat_think_flg:
      #     # think_value = self.chat.get_actor_think(actor, send_message, sys_prompt, input_images, temp_type, actor_name, user_name, sent_num, header_id_replace_flg, cfg.no_chat_key ,stop_event)
      #     think_value = self.chat.get_actor_think(actor, send_message, sys_prompt, input_img_set, input_vdo_set, temp_type, actor_name, user_name, sent_num, header_id_replace_flg, cfg.no_chat_key ,stop_event)
      #     if sys_prompt != "": sys_prompt += "\n\n"
      #     sys_prompt_t = sys_prompt + think_value

      # 最大返信トークン数
      max_new_tokens = actor.max_new_tokens

      # 途中終了
      if stop_event.is_set(): return None
      if self.check_msg_change(chk_message): return ""

      # ★チャットクラスからレスポンス取得
      response = self.chat.get_response_actor(actor, send_message, sys_prompt_t, input_img_set, input_vdo_set, max_new_tokens, actor.temperature, temp_type, actor_name, user_name, cfg.no_chat_key, sent_num, event_flg, no_print, name_no_msg_flg)

      if think_flg:
        if not response or response == "":
          # レスポンスが取得できなかったら再取得
          continue
        break

      if not response or response == "":
        # レスポンスが取得できなかったら再取得
        continue
      else:
        if not key_word:
          # キーワードが設定されていなかったら終了
          break
        else:
          # キーワードが設定されていたら、その言葉が入っているまで繰り返し
          key_words = key_word.split(",")
          find_flg = True
          for word in key_words:
            word_p = f"[Word: {word}]"
            if not word.strip() in response:
              find_flg = False
              break
            elif word_p in response:
              # [Word: ]形式を許可しない
              find_flg = False
              break
          if find_flg:
            break
          else:
            if try_num >= try_max:
              break

    e_time = time.time()

    # 途中終了
    if stop_event.is_set(): return None
    if self.check_msg_change(chk_message): return ""

    # 返信から特殊効果文字列を分ける
    res_eff_texts, res_simple = self.conv_effect_list(response)

    # 余計な空白を削除
    while "  " in res_simple:
      res_simple = res_simple.replace("  ", " ")
    while "\n\n" in res_simple:
      res_simple = res_simple.replace("\n\n", "\n")

    # 日付日時を追加
    if hasattr(cfg, "chat_datetime_flg") and cfg.chat_datetime_flg:
      res_eff_texts.append(get_datetime_chat())

    res_text = f"{actor_name}: {res_simple}"
    res_time = e_time - s_time
    time_text = f"<{res_time:.2f}s>"

    if not no_print:
      print(f"{res_text} {time_text}")

    # 途中終了
    if stop_event.is_set(): return None
    if self.check_msg_change(chk_message): return ""

    if actor_name:
      # 応答に名前が入っていたら削除
      if res_simple.startswith(f"{actor_name:}"):
        res_simple = res_simple[len(f"{actor_name:}:"):].strip()

    # 途中終了
    if stop_event.is_set(): return None
    if self.check_msg_change(chk_message): return ""

    if not think_flg:
      self.last_message = f"{actor_name}: {res_simple}"

      # 履歴に追加(+チャット欄更新なし) ※チャットランを更新しないのは文字を段階的に表示するため
      self.add_txt_chat(actor_name, res_simple, res_time=time_text, eff_texts=res_eff_texts, chat_update=False, new_line_num=new_line, font_tag=actor.chat_name)

    view_sec = cfg.chat_view_sec if hasattr(cfg, "chat_view_sec") else 0.1
    if hasattr(actor, "chat_view_sec"):
      view_sec = actor.chat_view_sec
      
    # チャット欄にキャラのレスポンスの効果を書き込む
    chat_response = res_simple
    if len(res_eff_texts) > 0:
      res_eff_value = ""
      for res_eff in res_eff_texts:
        if self.view_chat == 0:
          if "[Datetime:" in res_eff:
            continue
        res_eff_value += res_eff
      if res_eff_value != "":
        chat_response += " " + res_eff_value

    if not no_view:
      # キャラのレスポンスを画面に書き込む
      self.write_chat_actor(actor.chat_name, chat_response, time_text, new_line, view_sec=view_sec, stop_event=stop_event)

    # 思考のときは最後に全体に()を付ける。
    # if think_flg: chat_response = f"({chat_response})"
    # actor.is_chat_now = False

    return f"{actor_name}: {chat_response}"

  # 送信メッセージが変わっているかチェック
  def check_msg_change(self, message):
    if not message:
      # 思考中とイベント中はチャットを終了しない
      return False
    elif self.last_message != message:
      # メッセージが変わっていたらチャットを終了
      return True
    else:
      return False

  # チャット前状態復元
  def chat_restoration(self, rew_num=1, mode=1):
    self.send_menu_enabled(True)
    self.etr_msg.config(state=tk.NORMAL)
    self.btn_send.config(state=tk.NORMAL)
    self.etr_msg.insert("end", self.bef_message)

    # 履歴を1データ分削除
    self.chat.history_rewind(rew_num, mode)
    self.update_chat()
    self.bef_message = ""

  # 画像の説明文を取得
  def get_image_describe(self, actor, image_path):
    if not actor.eye_model:
      return None
    image = Image.open(image_path).convert("RGB")
    image_tensor = actor.eye_model.image_preprocess(image).to(actor.eye_model.device)

    user_text = "Describe the actoristics of the image in detail using bullet points." # 画像の特徴を箇条書きで詳細に書いて
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{user_text} ASSISTANT:"

    input_ids = actor.eye_tokenizer(text, return_tensors='pt').input_ids

    output_ids = actor.eye_model.generate(
        input_ids,
        max_new_tokens=256,
        images=image_tensor,
        use_cache=True)[0]

    return actor.eye_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

  # キャラクタのセリフを書き込む
  def write_chat_actor(self, chat_name:str, response, res_time=None, new_line=0, view_sec=0, stop_event=threading.Event()):
    if self.txt_chat.get("1.0", tk.END).strip() != "":
      new_line += 1 # 空白でないときは先頭に改行を加える

    while self.is_write_chat:
      time.sleep(0.1)

    self.is_write_chat = True

    self.write_chat_text(f"{chat_name}: ", new_line, font_tag=chat_name, req_see_flg=True)

    # レスポンスを段階的に表示
    # view_sec = self.cfg.chat_view_sec if hasattr(self.cfg, "chat_view_sec") else 0.1
    # view_sec = float(view_sec)

    if view_sec == 0:
      # 一度に全ての文を出力
      self.write_chat_text(response, font_tag=chat_name, req_see_flg=True)
    else:

      # [チャット履歴]非活性
      # self.menu_view_chat_enable(False)
      i = 0
      while i < len(response):
        # self.txt_chat.insert("end", add_text)
        self.write_chat_text(response[i], font_tag=chat_name, req_see_flg=True)
        time.sleep(view_sec)
        i += 1

    self.is_write_chat = False

      # チャットテキストの最後にシーク
      # self.txt_chat.see(tk.END)
    
  # 文章から感情を取得
  # def get_emotion(self, message, p_name):
  #   # if not self.classifier:
  #   #   return "Emotion: None"
  #   msg_emotion = self.classifier(message)
  #   add_emot_list = []
  #   for emot in msg_emotion[0]:
  #     if emot["label"] == "positive":
  #       add_emot_list.append(f"好き{emot['score']*100:.1f}%")
  #     elif emot["label"] == "neutral":
  #       add_emot_list.append(f"普通{emot['score']*100:.1f}%")
  #     elif emot["label"] == "negative":
  #       add_emot_list.append(f"嫌い{emot['score']*100:.1f}%")
  #   get_emot_value = f"[{p_name}の感情：{' ,'.join(add_emot_list)}]"

  #   return get_emot_value

  # 文字列の中に特定の文字列があるかチェック
  def check_pattern_in_string(self, pattern, target_string):
    match = re.search(pattern, target_string)
    return bool(match)

  # 画像の効果文字列を取得
  def get_image_eff_list(self, input_img_paths):
    img_eff_list = []
    for path in input_img_paths:          
      if path:
        if "[" in path or "]" in path:
          img_eff_list.append(f"<Image: {path}>")
        else:
          img_eff_list.append(f"[Image: {path}]")
    return img_eff_list

  # テキストの書き込みを制御
  def start_write_text(self, value, new_line_num=0, font_tag=None, req_see_flg=False):
    while self.is_write_chat:
      time.sleep(0.1)
    self.is_write_chat = True
    self.write_chat_text(value, new_line_num, font_tag, req_see_flg)
    self.is_write_chat = False

  # LoRAのリストをすべて取得
  def get_lora_list(self, actor):
    lora_path = actor.full_lora_path
    if not lora_path:
      return None
    if not hasattr(actor, "lora_adapter_name"):
      return None

    lora_list = []
    for entry in os.listdir(lora_path):
      full_path = os.path.join(lora_path, entry)
      if os.path.isdir(full_path) and entry.startswith(actor.lora_adapter_name):
        lora_list.append(full_path)

    # フォルダ名から数値を取り出して、数値順にソート
    lora_list = sorted(lora_list, key=lambda x: int(x.split('-')[1]))

    return lora_list

  # チャット画面をコマンド画面に変更or戻す
  def change_command_view(self, view_flg=True):
    if view_flg:
      self.txt_chat.config(state=tk.NORMAL)
      self.txt_chat.delete(1.0, tk.END)
      self.txt_chat.config(state=tk.DISABLED)
      self.txt_chat.config(fg="white", bg="black", font=self.cfg.font_cmd)
      self.etr_msg.config(state=tk.DISABLED)
      self.btn_send.config(state=tk.DISABLED) # この画面は中断もできない
      self.command_view_flg = True

    else:
      self.txt_chat.config(state=tk.NORMAL)
      self.txt_chat.delete(1.0, tk.END)
      self.txt_chat.config(state=tk.DISABLED)
      self.txt_chat.config(fg="black", bg=self.cfg.color_txt_chat, font=self.cfg.font_chat)
      self.etr_msg.config(state=tk.NORMAL)
      self.btn_send.config(state=tk.NORMAL)
      self.command_view_flg = False

  # チャットキャンバスを元に戻す
  def return_cnv_chat(self):
    if self.graph_view_flg:
      self.cnv_chat.destroy()
      self.cnv_chat = tk.Canvas(self.panel_03_2, bg="gray40", highlightthickness=1)
      self.cnv_chat.pack(fill=tk.BOTH, expand=True)
      self.graph_view_flg = False

  # 送信関連コントロールの活性/非活性
  # def send_control_activ(self, enable=True):
  #   if enable:
  #     self.btn_send.config(state=tk.NORMAL)
  #     self.etr_msg.config(state=tk.NORMAL)
  #   else:
  #     self.btn_send.config(state=tk.DISABLED)
  #     self.etr_msg.config(state=tk.DISABLED)

  # メッセージ送信時のコントロールの活性・非活性
  def send_menu_enabled(self, enable_flg=True, train_flg=False):
    # mode = 1
    
    if enable_flg:
      tk_state = tk.NORMAL
      text=BUTTON_SEND
    else:
      tk_state = tk.DISABLED
      # if mode == 1:
      #   # チャットEXは追加で送信もできる
      #   text=BUTTON_SEND_STOP
      # else:
      #   text=BUTTON_STOP
      text=BUTTON_SEND_STOP

    if train_flg:
      # 学習時は[送信]のまま非活性
      self.btn_send.config(state=tk_state)
    else:
      # ボタンの[送信][中断]切り替え
      self.btn_send.config(text=text)

    # メッセージ
    # if mode != 1 or tk_state == tk.NORMAL:  
    #   self.etr_msg.config(state=tk_state)

    self.menu_file.entryconfig(MENU_CHAT_CLEAR, state=tk_state)
    self.menu_file.entryconfig(MENU_ACTOR_READ, state=tk_state)
    self.menu_file.entryconfig(MENU_ACTOR_RELEASE, state=tk_state)
    # self.menu_file.entryconfig(MENU_IMAGE_READ, state=tk_state)
    # self.menu_file.entryconfig(MENU_IMAGE_CLEAR, state=tk_state)
    self.menubar.entryconfig(MENU_EDIT, state=tk_state)
    # self.menubar.entryconfig(MENU_MODE, state=tk_state)
    # self.menubar.entryconfig(MENU_TOOL, state=tk_state)

  # メニューチャット履歴を活性/非活性にする
  # def menu_view_chat_enable(self, enable_flg=True):
  #   if enable_flg:
  #     tk_state = tk.NORMAL
  #     self.btn_send.config(state=tk_state)
  #   else:
  #     tk_state = tk.DISABLED

  # キャラが有効かどうかでコントロールの活性・非活性
  def actor_menu_enable(self, enable_flg=True):
    if enable_flg:
      tk_state = tk.NORMAL
    else:
      tk_state = tk.DISABLED

    self.etr_msg.config(state=tk_state)
    self.btn_send.config(state=tk_state)
    self.menu_file.entryconfig(MENU_ACTOR_RELEASE, state=tk_state)
    self.menubar.entryconfig(MENU_EDIT, state=tk_state)

  # 辞書形式の履歴から文字列を取得
  def get_dic_to_text(self, dic):
    text = ""
    if dic["name"]:
      text = f"{dic["name"]}: "
    text += dic["text"]

    if dic["effects"] and len(dic["effects"]) > 0:
      for eff in dic["effects"]:
        if self.view_chat == 0:
          if eff.startswith("[Datetime:") \
            or eff.startswith("<Image:") \
            or eff.startswith("[Image:"):
            continue

        if text != "":
          text += " "
        text += eff

    return text

  # 履歴を画面表示用に変換
  def history_list_to_text(self):
    chat_text = ""
    for dic in self.chat.history_dic_list:
      # 新しい履歴からテキスト作成
      text = self.get_dic_to_text(dic)
      if chat_text != "":
        chat_text += "\n"
      chat_text += text
  
    return chat_text
  
  # チャット画面を更新
  def update_chat(self):
    if self.command_view_flg:
      return

    self.txt_chat.config(state=tk.NORMAL)
    self.txt_chat.delete("1.0", "end")
    
    # 新しい履歴
    chat_value = self.history_list_to_text()

    # 会話間隔
    # if mode == 2 or mode == 3 or mode == 4:
    #   new_line = self.cfg.new_line_num_dis if hasattr(self.cfg, "new_line_num_dis") else 0
    # else:
    
    new_line = self.cfg.new_line_num if hasattr(self.cfg, "new_line_num") else 0
    
    if new_line > 1:
      chat_value_line = chat_value.split("\n")
      new_line_str = "\n" * (new_line + 1)
      chat_value = new_line_str.join(chat_value_line)

    self.txt_chat.insert("end", chat_value)

    user_name = self.cfg.user_name
    actor_names = [c.chat_name for c in self.actors if c]
    other_names = cfg.other_names

    if chat_value != "":
      # フォントの文字色を変える
      if self.cfg.chat_color_flg:
        self.change_line_color(user_name, self.cfg.chat_color, actor_names+other_names)
        for actor in self.actors:
          if actor:
            self.change_line_color(actor.chat_name, actor.chat_color, [user_name]+other_names)

        # その他の名前も色を変える
        if hasattr(cfg, "other_names") and hasattr(cfg, "other_colors"):
          for other_name, other_color in zip(cfg.other_names, cfg.other_colors):
            self.change_line_color(other_name, other_color, actor_names+[user_name]+other_names)

    self.txt_chat.see(tk.END)

    self.txt_chat.config(state=tk.DISABLED)

    # # ふきだしチャットのアップデート
    # if self.bub_chat:
    #   self.bub_chat.chat_update()

  # チャット履歴に書き込んでチャットテキストに文字列を追加
  def add_txt_chat(self, name, text, eff_texts=None, res_time=None, chat_update=False, new_line_num=0, font_tag=None):
    text = text.strip()
    if not name and text == "":
      return    
    name_text = f"{name}: {text}" if name else text
    
    if self.txt_chat.get("1.0", "end").strip() != "":
      name_text = "\n" + name_text

    text_eff_time = text_eff = text_view = name_text

    eff_str_s = ""
    eff_str = ""
    if eff_texts and len(eff_texts) > 0:
      for eff in eff_texts:
        eff_str += " " + eff
        if not eff.startswith("[Datetime:") and not eff.startswith("<Image:") and not eff.startswith("[Image:"):
          eff_str_s += " " + eff
    
    if eff_str != "":
      text_eff += eff_str
    # if eff_str_s != "":
    #   text_view += eff_str_s
    # text_eff_time = text_eff
    # if not res_time is None:
    #   text_eff_time += " " + res_time

    # Debug
    if "  " in text_eff:
      print(text_eff)
      text_eff.replace("  ", " ")

    # self.chat.view_history_simple += name_text
    # self.chat.view_history += text_eff
    # self.chat.view_history_view += text_view
    # self.chat.view_history_time += text_eff_time

    self.mem_output_flg = False

    self.chat.history_dic_list.append({"name": name, "text": text, "effects": eff_texts, "response": res_time})

    log_text_eff_time = text_eff_time.replace("\n","")

    print("*** Message/Response with time " + "*"*69 + f"\n* {log_text_eff_time}\n" + "*"*100)
    logger.info("*** Message/Response with time ***\n" + log_text_eff_time)

    # logger.info("*** Full History Log ***\n" + self.chat.view_history)

    # if self.var_mode.get() == 0:
    if self.view_chat == 0:
      add_text = name_text
    elif self.view_chat == 1:
      add_text = text_eff

    # elif self.var_view_chat.get() == 2:
    #   add_text = text_eff
    # else:
    #   add_text = text_eff_time
    # # チャットEX
    # eff_list_s = [eff for eff in eff_texts if ]
    # if self.var_view_chat.get() == 0:
    #   add_text = f"{name}: {text}" 
    # elif self.var_view_chat.get() == 1:
    #   add_text = f"{name}: {text}" 
    # elif self.var_view_chat.get() == 2:
    #   add_text = text_eff
    # else:
    #   add_text = text_eff_time

    if chat_update:
      self.start_write_text(add_text, new_line_num, font_tag=font_tag, req_see_flg=True)

    self.txt_chat.see(tk.END) # 一番下を表示
    self.master.update()

  # チャットテキストに文字列を追加(req_see_flg:強制スクロールフラグ)
  def write_chat_text(self, value, new_line_num=0, font_tag=None, req_see_flg=False):
    try:
      # 自動スクロールの有無(一番下のとき)
      if req_see_flg:
        is_auto_see = True
      else:
        is_auto_see = True if self.txt_chat.yview()[1] == 1.0 else False

      is_lock = self.txt_chat['state'] == tk.DISABLED
      self.txt_chat.config(state=tk.NORMAL)

      for _ in range(new_line_num):
        value = "\n" + value # 改行追加

      if self.cfg.chat_color_flg and font_tag is not None:
        self.txt_chat.insert(tk.END, value)
        end_index = float(self.txt_chat.index(tk.END))
        start_index = end_index - 1.0
        self.txt_chat.tag_add(font_tag, str(start_index), f"{end_index}-1c")
      else:
        self.txt_chat.insert(tk.END, value)
        
      if is_lock: # ロックがかかっていたときのみロックをかける
        self.txt_chat.config(state=tk.DISABLED)

      if is_auto_see:
        # 最終行を表示
        self.txt_chat.see(tk.END)
    except Ellipsis as e:
      pass

  # チャットの特定の文字から始まる行の色を変える
  def change_line_color(self, chat_name, font_color=None, stop_names=None):
    if not font_color:
      return

    no_chat_key = self.cfg.no_chat_key
    chat_split_key = self.cfg.chat_split_key
    find_flg =  False
    # 行数を取得
    total_lines = int(self.txt_chat.index('end-1c').split('.')[0])  # 全行数を取得

    # 全行をループして、各行のテキストを取得
    for line_num in range(1, total_lines + 1):
      # 行ごとのテキストを取得（'line_num.0' から 'line_num.end'）
      line_start = f"{line_num}.0"
      line_end = f"{line_num}.end"
      line_text = self.txt_chat.get(line_start, line_end)

      if line_text.startswith(f"{chat_name}:"):
        find_flg = True
      else:
        if line_text.startswith(no_chat_key):
          find_flg = False
        elif line_text.startswith(chat_split_key):
          find_flg = False
        elif stop_names:
          for s_name in stop_names:
            if line_text.startswith(f"{s_name}:"):
              find_flg = False
              break
        else:
          find_flg = False

      if find_flg:
        self.txt_chat.tag_add(chat_name, line_start, line_end)

    # テキスト全体の範囲を指定
    # start_index = "1.0"
    # end_index = tk.END
    
    # # 行ごとに処理
    # while True:
    #   # line_start = self.txt_chat.search(search_str, start_index, stopindex=end_index, regexp=True)
    #   line_start = self.txt_chat.search(f"^{chat_name}:", start_index, stopindex=end_index, regexp=True)

    #   if line_start:
    #     find_flg = True
    #   else:
    #     for s_name in stop_names:
    #       line_start_other = self.txt_chat.search(f"^{s_name}:", start_index, stopindex=end_index, regexp=True)
    #       if line_start_other:
    #         # もし他の名前が見つかったら色塗り中止
    #         find_flg = False

    #   if not line_start:
    #     break
    #   if find_flg:
    #     line_end = self.txt_chat.search("\n", line_start, stopindex=end_index)

    #     # タグを設定して文字色を変更
    #     # tag_name = f"{chat_name}"
    #     # self.txt_chat.tag_configure(tag_name, foreground=color)
    #     self.txt_chat.tag_add(chat_name, line_start, line_end)

    #   # 次の行の開始位置を更新
    #   start_index = f"{line_end}+1c"

  # 開始時のキャラクターのロード
  def start_load_actor(self, actor_path):
    self.actors = [None] # 一人分の枠を確保
    ret = False

    if hasattr(self.cfg, "actor_auto_load") and self.cfg.actor_auto_load:
      ret = self.load_actor(actor_path, c_id=0, init_flg=True)
    self.actor_menu_enable(ret)

  # キャラクターをロード
  def load_actor(self, actor_path, c_id=0, info_only=False, init_flg=False) -> bool:
    while len(self.actors)-1 < c_id:
      self.actors.append(None)

    if not os.path.isdir(actor_path):
      self.actors[c_id] = None
      return False
    try:
      if not info_only:
        self.actors[c_id] = actor(actor_path, self.cfg)
        self.actors[c_id].owner = self

      # メインキャラ画像をセット
      # self.set_main_img(c_id)
      self.actors[c_id].load_main_img()

      # メインキャラのみ変更する項目
      if c_id == 0:
        # プロフィールをセット(メインキャラのみ)
        self.txt_prof.config(state=tk.NORMAL)
        self.txt_prof.delete("1.0", tk.END) # 全削除
        self.txt_prof.insert("1.0", self.actors[c_id].get_profile())
        self.txt_prof.config(state=tk.DISABLED)

      # return
      if self.actors[c_id].model_enable:
        if self.cfg.thread_model_load and not info_only:
          # キャラクタロード時にモデルを別スレッドでロード

          cuda_num = self.cfg.def_cuda_num if c_id != 1 else self.cfg.def_sub_cuda_num

          self.load_stop_event = threading.Event()
          self.load_thread = threading.Thread(target=self.actors[c_id].load_model, args=(None, None, None, False, cuda_num, self.load_stop_event), daemon=True)
          self.load_thread.start()

      title = f"{WINDOW_TITLE} - {self.actors[0].name}"
      if self.actors[0].name != os.path.basename(self.actors[0].path):
        title += f" ({os.path.basename(self.actors[0].path)})"

      for i in range(c_id):
        title += f" - {self.actors[i+1].name}"
        if self.actors[i+1].name != os.path.basename(self.actors[i+1].path):
          title += f" ({os.path.basename(self.actors[i+1].path)})"

      self.master.title(title)

      # メイン画像作成
      self.main_img = self.create_main_img()

      # テキストボックスにタグ付け
      self.txt_chat.tag_configure(self.actors[c_id].chat_name, foreground=self.actors[c_id].chat_color)

      self.after(10, lambda: self.open_window(self.actors[0]))
      return True
    except Exception as e:
      print(f"Error: {str(e)}")
      print(f"Error: Failed to Load actor.")
      self.actors[c_id] = None
      return False

  # メイン画像取得
  def create_main_img(self):
    image = None

    if not self.actors is None and not self.actors[0] is None \
       and not self.actors[0].main_img is None:

      image = self.actors[0].main_img.copy()
      if len(self.actors) > 1:
        # サブキャラ画像と統合
        img_w, img_h = image.size
        sub_w = int(img_w/3)
        ws = int(img_w/50)
        wn = 0
        for i in range(len(self.actors)-1):
          sub_img = self.actors[i+1].main_img
          if not sub_img is None:
            sub_h = int(sub_img.size[1]*sub_w/sub_img.size[0])
            sub_img = sub_img.resize((sub_w, sub_h), Image.Resampling.LANCZOS)
            sub_x = img_w - sub_w - wn - ws
            sub_y = img_h - sub_h - ws

            # 画像を埋め込む
            image.paste(sub_img, (sub_x, sub_y))

            # 境界線
            draw = ImageDraw.Draw(image)
            draw.rectangle((sub_x-1, sub_y-1, sub_x+sub_w, sub_y+sub_h), outline=(255, 255, 255))
            wn += sub_w + sub_w

    return image #, image2

  # キャンバスに画像を描画
  def draw_canv_img(self, image, canvas, size_w=None, size_h=None, image2=None):
    if not image:
      return

    if not size_w:
      width = canvas.winfo_width()
    else:
      width = size_w

    if not size_h:
      height = canvas.winfo_height()
    else:
      height = size_h

    img_w, img_h = image.size

    if width * img_h/img_w < height:
      height = int(width * img_h/img_w)
      x = 0
      y = int((canvas.winfo_height() - height)/2)
    else:
      width = int(height * img_w/img_h)
      x = int((canvas.winfo_width() - width)/2)
      y = 0

    if not image is None and width > 0 and height > 0:
      imgtk = self.resize_image(image, width, height)
      if image2:
        imgtk2 = self.resize_image(image2, width, height)

      canvas.delete("all")

      canvas.create_image(x, y, anchor=tk.NW, image=imgtk)
      if image2:
        canvas.create_image(x, y, anchor=tk.NW, image=imgtk2)

    else:
      imgtk = None

    return imgtk

  # 画像入力があるモデルのときは取得
  def get_input_image(self, img_max_size=-1, path_get_flg=True):

    input_images = []
    input_img_paths = []

    if self.win_picture:
      if len(self.win_picture.image_dic_list) > 0:
        
        for dic in self.win_picture.image_dic_list:
          image = dic["data"]
          path = dic["path"]
         
          # if not path and path_get_flg:
          #   # 画像を出力してパスを取得(思考時以外)
          #   path = self.win_picture.output_image()

          if path and not os.path.isfile(path):
            path = None # パスのファイルが見つからなかったら
          if image and img_max_size > 0:
            # 大きい画像はリサイズ
            image = self.resize_input_image(image, img_max_size)

          input_images.append(image)
          input_img_paths.append(path)

    return input_images, input_img_paths
  
  # メイン画像のリサイズ(TKイメージ)
  def resize_image(self, image_tk, width, height):
      # 画像のアスペクト比を保持しながらサイズを変更
      image = image_tk
      ratio = image.width / image.height
      if width / ratio <= height:
          new_width = width
          new_height = width / ratio
      else:
          new_width = height * ratio
          new_height = height

      resized_image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)

      return ImageTk.PhotoImage(resized_image)

  # フォルダの中のファイルの数
  def count_files(self, dir_path):
    # 指定されたフォルダ内のファイルの数を取得
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])


if __name__ == "__main__":
   # ArgumentParserオブジェクトを作成
    parser = argparse.ArgumentParser(description='AI Viewer')

    # コマンドライン引数の定義
    parser.add_argument('model_path', nargs='?', default=None, help='Inport AI Model Directory Path')

    # 引数の解析
    args = parser.parse_args()
    cfg = Config(CONFIG_FILE)
    if not cfg:
      print(f"Error: Get Config File. [{str(Path(CONFIG_FILE))}]")
      exit()

    print(args)

    root = TkinterDnD.Tk()
    app = Application(master=root, args=args, cfg=cfg)
    app.mainloop()