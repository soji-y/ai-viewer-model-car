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

    # [setting]辞書のキーと値をクラスメンバとして設定
    for key, value in cfg_settings.items():
        setattr(self, key, value)

    # ここで設定しておいた方が良いものを設定
    if not hasattr(self, "img_redraw_msec"): self.img_redraw_msec = 100 # メイン画像再描画時間(ミリ秒)
    if not hasattr(self, "chat_hist_flg"): self.chat_hist_flg = True # チャット履歴
    if not hasattr(self, "max_sentences"): self.max_sentences = -1 # チャット履歴なしの文数
    if not hasattr(self, "max_sentences_hist"): self.max_sentences_hist = 2 # チャット履歴ありの文数

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

    self.win_char_view_sec = 0.3 # チャット表示間隔
    
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
    self.owner = None  # 起動アプリ

    # 他スレッドでチェックされるので早めに設定しておく
    self.model = None # Transformers
    self.tokenizer = None

    self.model_loading_flg = False # モデル読み込み中フラグ
    self.load_lora_flg = False # LoRA読み込みフラグ

    # モデルコンフィグファイルを読み込む
    self.load_config()

  # コンフィグファイルを読み込む
  def load_config(self, no_param_flg=False):
    try:
      actor_cfg_path = os.path.join(self.path, self.cfg.actor_data)

      if not os.path.isfile(actor_cfg_path):
        print_log(f"Error: Not Fonnd! [{actor_cfg_path}]", 1)
        return
      
      data=None
      # モデル設定ファイルを読み込む
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

      # モデルのフルパス
      self.full_model_path = os.path.join(self.path, self.model_path)

      # LoRAの設定
      self.lora_path = None
      if "lora" in data:
        actor_lora = data["lora"]
        if "path" in actor_lora:
          self.lora_path = actor_lora["path"]
          self.full_lora_path = os.path.join(self.path, self.lora_path) if self.lora_path != "" else None
        if "adapter_name" in actor_lora:
          self.lora_adapter_name = actor_lora["adapter_name"]
        if "bf16" in actor_lora:
          self.lora_bf16 = actor_lora["bf16"]
        if "bits" in actor_lora:
          self.lora_bits = actor_lora["bits"]

      actor = data["actor"] # [必須]
      self.name = actor["name"] # [必須]

      # パラメータ(その他)：辞書のキーと値をクラスメンバとして設定
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

      self.chat_remenb_num = 10 # チャットで記憶できる会話数
      self.def_picture = False

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
    
    if self.model_loading_flg:
      # モデル読み込み中の時は待機
      while self.model_loading_flg:
        time.sleep(0.1)
      # この時点で読み込みは完了してるので、必ずself.loaded_model_nameが設定されている。
      if not self.model_name is None and self.loaded_model_name == self.model_name:
        # 一つ前に読み込んだモデル名が現在のモデル名と一致していれば終了。
        return

    # modelとtokenizerを完全に開放する
    self.release_model()
    time.sleep(0.5)

    self.model_loading_flg = True # ロード中
    self.model_merged_flg = False # マージ済みか
    try:
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

        # モデルの読み込み
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

        elif "idefics2" in model_name:
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
          self.model.config.is_merged = True

        else:
          # それ以外のモデル
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
          print_log(f"Failed to Load Model!", log_type=1)
          self.model_loading_flg = False
          return

        if not self.model:
          print_log(f"Loaded cTransrate2 Model! [{str(Path(self.full_merge_path_num))}]")
        else:
          print_log(f"Loaded Model! [{model_name}]")

      # 途中終了
      if stop_event and stop_event.is_set():
        self.model_loading_flg = False
        return

      self.load_lora_flg = False

      if not lora_dir_path:
        # LoRAが存在しないときは読み込みなし
        print_log("Without using LoRA.")
      else:
        # Loraあり
        if self.last_lora_dir_path:        
          if "GPTQ" in self.model_type:
            self.model = self.model.to(torch.float16)

          self.model = PeftModel.from_pretrained(self.model, self.last_lora_dir_path, torch_dtype=torch.float16, device_map="auto", load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit) #{'': 0})

          # LoRAのマージフラグ(先にeval()をするためここではマージしない)
          self.model.config.is_merged = True

          self.load_lora_flg = True
          print_log(f"Loaded LoRA-Adapter! [{self.last_lora_dir_path}]")

        else:
          # Loraなし
          print_log("Without using LoRA.")

      if self.model:
        # 推論モードに変更
        self.model.eval()

        if self.model.config.is_merged:
          if self.merge_flg:
            # LoRAとマージ
            self.model = self.model.merge_and_unload()
            self.model_merged_flg = True

      if not self.model is None:
        self.loaded_model_name = self.model_name # 読み込んだモデル名を保存

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
    if not self.tokenizer is None:
      self.tokenizer = None

    if self.cfg.device.type == "cuda":
      torch.cuda.empty_cache()
    gc.collect()

  # モデル画像を読み込み
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


  # 画面用のプロフィールを取得
  def get_profile(self):

    prof_lines = []

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

  # イベントが発生するか
  def occur_event(self):
    if random.random() < self.action_ratio:
      self.is_occur_action = True


# チャットクラス
class Chat():
  def __init__(self):
    self.history_dic_list:list = []   # フル入力のチャット履歴(リスト)

    self.bef_prompt = "" # 前回の入力プロンプト
    self.bef_response = "" # 前回のレスポンス

    self.print_log = None

  # [チャット] 送信メッセージを編集して受け取る
  def get_response_actor(self, actor, send_message, sys_prompt, image_set, video_set, max_tokens, temperature, temp_type, user_name, no_chat_key, sent_num, direct_flg=False, no_print=False, name_no_msg_flg=False):
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
          add_generation_prompt=True
      )
    else:
      text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
      )

    return text, messages

  # チャット履歴を巻き戻す
  def history_rewind(self, rew_num=1) -> bool: 
    history_dic_list = self.history_dic_list

    if len(history_dic_list) == 0:
      return False

    if len(history_dic_list) < rew_num:
      rew_num = len(history_dic_list)

    self.history_dic_list = history_dic_list[:-rew_num]

    return True

  # 基本チャットレスポンス取得
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

    self.bef_prompt = prompt # 前回のプロンプト
    response = ""
    
    if not no_log:
      logger.info("*** Full prompt ***\n" + prompt)

    try:
      with torch.no_grad():
        if model is not None:
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

            output_ids = model.generate(
                **token_ids,
                max_new_tokens=max_tokens,
                repetition_penalty=actor.repetition_penalty, # 繰り返しを制限(1.0だと制限なし)
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

    self.load_img = None
    self.item_img = None
    self.actor_img = None
    self.chat_img = None
    self.chat = None

    # ウィンドウ
    self.win_picture = None

    # デフォルト値を設定
    self._init_default()

    # メイン画面の再描画時間
    self.redraw_msec = cfg.img_redraw_msec

    # ランダムシード値
    random.seed(set_random_seed(cfg, RANDOM_SEED))

    master.fonts = cfg.font_form
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

    # モデル画像用キャンパスを作成
    self.cnv_actor = tk.Canvas(self.panel_02_1, width=cfg.actor_win_size[0], height=cfg.actor_win_size[1], bg="gray20")

    # プロフィール用テキストを作成
    self.txt_prof = tk.Text(self.panel_02_2, width=0, fg="white", bg="gray20")

    # チャット内容用テキストボックスを作成
    self.scrollbar = tk.Scrollbar(self.panel_03_1)  # Scrollbarを作成
    self.txt_chat = tk.Text(self.panel_03_1, height = 150,font=cfg.font_chat, fg="black", bg=self.cfg.color_txt_chat, undo=True, highlightthickness=1, yscrollcommand=self.scrollbar.set)
    self.txt_chat.bind('<Control-Shift-Key-Z>', self.redo) # Redoをバインド

    self.etr_msg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.btn_send.pack(side=tk.RIGHT)
    self.cnv_actor.pack(fill=tk.BOTH, expand=True)
    self.txt_prof.pack(fill=tk.BOTH, padx=1, expand=True)
    self.txt_prof.config(state=tk.DISABLED)  # テキストボックスを非活性にする
    self.txt_prof.config(font=cfg.font_prof)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.txt_chat.pack(fill=tk.BOTH, expand=True)

    self.txt_chat.config(state=tk.DISABLED)  # テキストボックスを非活性にする

    # ScrollbarとTextコントロールを連携
    self.scrollbar.config(command=self.txt_chat.yview)

    # チャットクラス
    self.chat = Chat()
    self.chat.owner = self
    self.chat.print_log = print_log
    self.etr_msg.focus_set()

    # チャット書き込み中フラグ
    self.is_write_chat = False

    # キーEnterイベントを設定
    self.etr_msg.bind("<Return>", self.press_key)

    # ウィンドウの閉じるボタンが押されたときの処理
    self.master.protocol("WM_DELETE_WINDOW", self.menu_close)

    # 入力画像ウィンドウ
    self.win_picture = None

    # モデルの画像
    self.main_img = None
    self.actors = [None]

    self.bef_message = ""
    self.last_message = "" # 保存(この値が変わっていたら前回の送信キャンセル)

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

    # 画像表示(ループ)
    self.master.after(0, lambda: self.actor_img_update())

    # ドラッグ＆ドロップの設定
    self.master.drop_target_register(DND_FILES)
    self.master.dnd_bind('<<Drop>>', self.on_drop)

  # 初期化中のデフォルト値
  def _init_default(self):
    self.chat_edit_flg = False
    self.load_img = None
    self.chat_thread = None
    self.load_thread = None

    self.chat_stop_event = threading.Event()
    self.load_stop_event = threading.Event()
    self.chat_stop_event.clear()

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
      pass

  # メニューのラベルを変更
  def change_menu_item_name(self, menu_item, old_name, new_name):
    try:
      index = menu_item.index(old_name)  # ラベル名を使用してインデックスを取得
      menu_item.entryconfig(index, label=new_name)  # インデックスを使用して名前を変更
    except ValueError:
      print(f"Error: Label [{old_name}] was not found.")

  # [メニュー]チャットのクリア
  def menu_clear_chat(self):
    if len(self.chat.history_dic_list) == 0:
      return

    result = messagebox.askyesno(MENU_CHAT_CLEAR, f"チャット内容をクリアしますか？")
    if not result:
      return

    self.clear_chat()

  # [メニュー]モデル読込
  def menu_load_actor(self, dir_path=False):
    actor = self.actors[0]

    if dir_path:
      actor_dir_path = dir_path
    else:
      open_dir = None
      if os.path.isdir(self.actor_path):
        open_dir = self.actor_path

      actor_dir_path = filedialog.askdirectory(initialdir=open_dir)

    req_files = [self.cfg.actor_data]
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

    # チャットスレッドの終了を待つ
    if self.chat_thread and self.chat_thread.is_alive():
      print("Chat-Thread is still Running. Forcefully Terminating...")
      # スレッドに停止の合図を送る
      self.chat_stop_event.set()
      self.chat_thread.join()

    self.actors[0].release_model()
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

      ret = self.actors[0].load_config(no_param_flg=True)
      self.open_window(self.actors[0])

      if ret:
        # モデルを別スレッドでロード
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

    self.chat_thread = threading.Thread(target=self.send_message_chat, args=([name, message, self.actors, 1, self.chat_stop_event, True, False]), daemon=True)
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

    result = messagebox.askyesno(MSG_TTL_CONFIR, f"チャットテキスト編集しますか？\n※チャット開始時に編集内容が確定されます。")
    if not result:
      return

    self.chat_edit_flg = True

    # チャット表示
    self.view_chat = 1
    self.update_chat()

    self.txt_chat.config(state=tk.NORMAL, bg="white")

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

    target = self.send_message_chat
    args = [None, message, self.actors, 1, self.chat_stop_event, False, False]

    # 入力メッセージのエントリーの内容を削除
    self.etr_msg.delete(0, "end")
    
    self.chat_stop_event.clear()
    self.chat_thread = threading.Thread(target=target, args=(args), daemon=True)
    self.chat_thread.start()

  # キープレスイベント(Enter)
  def press_key(self, _):
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
            _, input_img_paths = self.get_input_image(img_max)
            
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

    # 履歴用
    new_history_dic_list = [] 
    new_history_dic = {}
    text_list = []
    eff_list = []
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

        # 履歴への追加
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
            text_list, eff_list = [], []
        if tmp_name:
          bef_name = tmp_name
        elif tmp_name == "":
          bef_name = ""

        text_simple = text
        text_view = text

        # 正規表現を使用して指定の範囲を空文字列に置換
        for value in EFFECT_STRING:
          pattern = re.compile(fr"\[{value}:.*?\]")
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

          # 履歴
          if value == "Image" and "<Image:" in tmp_text:
            pattern_2 = re.compile(fr"\<Image:.*?\>")
            matches_2 = re.findall(pattern_2, tmp_text)
            eff_list.extend(matches_2)
            tmp_text = re.sub(pattern_2, '', tmp_text).strip() # 削除

          matches = re.findall(pattern, tmp_text)
          eff_list.extend(matches)
          tmp_text = re.sub(pattern, '', tmp_text).strip() # 削除

        text_list.append(tmp_text)

      # 最後の一行
      new_history_dic["name"] = bef_name if bef_name != "" else None
      new_history_dic["text"] = "\n".join(text_list)
      new_history_dic["effects"] = eff_list
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
    if not num:
      n = 2
    else:
      n = num

    ret = self.chat.history_rewind(rew_num=n)
    if ret:
      self.update_chat()

    return ret

  # 画像を更新(ループ処理)
  def actor_img_update(self):
    try:
      canvas = self.cnv_actor
      
      if not self.main_img:
        canvas.delete("all")
      else:
        # リサイズ後の画像を取得
        self.actor_img = self.draw_canv_img(self.main_img, canvas)

    except Exception as e:
      self.cnv_actor.delete("all")

    self.master.after(self.redraw_msec, lambda: self.actor_img_update())

  # チャットのクリア
  def clear_chat(self):    
    self.chat.history_dic_list.clear()
    self.update_chat()
    
  # 画像のクリア
  def menu_clear_image(self):
    del self.load_img
    self.load_img = None
    self.load_img_tags = None
    self.master.update()

  # [チャット] メッセージを送信して返信を得る
  def send_message_chat(self, msg_name, message:str, actor_list:list, res_count=1, stop_event:threading.Event=None, retake=False, start_flg=False):

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

          # チャットの内部処理：エラーでも中断でも後処理は必要なので別スレッドにする
          actor.is_chat_now = True
          result = self.chat_proc(self.cfg, actor, msg_name, message, None, stop_event, retake, start_flg)
          actor.is_chat_now = False

          message = "" 
                   
          # メッセージを履歴に書き込み後に途中終了されていたら履歴を戻す
          if result is None:
            self.chat_restoration(rew_num=i+1)

          actor.last_action_time = datetime.datetime.now()

    except Exception as e:
      print("Failed to Get Chat Response.")
      print(f"Error: {str(e)}")
      traceback.print_exc()

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

    # モデル有効
    if actor.model_enable:
      if not actor.model:
        self.status_bar_msg("モデルを読み込んでいます...")
        actor.load_model(stop_event=stop_event)
    if not actor.model:
      print_log("Error: Failed to Load Model.", log_type=1)
      return False
        
    return True

  # [チャット] チャット処理内部
  def chat_proc(self, cfg:Config, actor:actor, msg_name, message, set_sys_prompt=None, stop_event:threading.Event=None,  retake=False, start_flg=False) -> Optional[str]:
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

    # システムプロンプトを取得
    actor_name = actor.chat_name

    # 最大文数
    if actor.max_sentences:
      sent_num = actor.max_sentences
    else:
      sent_num = cfg.max_sentences if not cfg.chat_hist_flg else cfg.max_sentences_hist

    # 会話間隔
    if start_flg:
      new_line = 0
    else:
      new_line = cfg.new_line_num if hasattr(cfg, "new_line_num") else 0

    name_no_msg_flg = self.prop.get_property(actor, "name_no_msg_flg", False) 
    eff_texts = []    
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

    my_msg = tmp_message
    if not retake:
      # リテイクのときはそのまま
      if not tmp_message is None:
        if tmp_message == "":
          if len(eff_texts) == 0:
            my_msg = f""
        else:
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
      eff_texts = [item for item in eff_texts if not datetime_ptn.search(item)] # 時刻
      eff_texts = [item for item in eff_texts if not img_path_ptn.search(item)] # 画像
      eff_texts.append(get_datetime_chat())

    # 途中終了
    if stop_event.is_set(): return None

    top_msg =""
    if cfg.chat_hist_flg:

      # こっちを取得しないと履歴が途中までしか取得できない可能性あり
      talk_history_list = [f"{dic['name']}: {dic['text']}" if dic["name"] else f"{dic['text']}" for dic in self.chat.history_dic_list if dic["name"] or (dic["text"] and not dic["text"].startswith(cfg.chat_split_key))]

      len_hist = len(talk_history_list)
      if len_hist > actor.chat_remenb_num:
        talk_history_list = talk_history_list[len_hist-actor.chat_remenb_num:]

      # [Datetime:]部分を削除
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

    # [Datetime:][Image:]はメッセージに入れない
    eff_texts_s = [item for item in eff_texts if not datetime_ptn.match(item) and not img_path_ptn.match(item)]
    send_message += "" if len(eff_texts_s) == 0 else "\n" + " ".join(eff_texts_s)

    # 入力画像とパスを入力画像ウィンドウから取得
    input_images, input_img_paths = self.get_input_image(actor.input_image_max)
    input_img_set = [input_images, input_img_paths]
 
    # 動画
    input_vdo_set = [None, None]

    # 画像パスを追加
    if len(input_img_paths) > 0:
      eff_texts.extend(self.get_image_eff_list(input_img_paths))

    chk_message = f"{user_name}: {my_msg}"
    self.last_message = chk_message

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

    # 送信メッセージ(全文)を表示
    send_message_print = '\n'.join(['* ' + s for s in send_message.split("\n")])
    print(f"*** Full Send Message " + "*"*78 + f"\n{send_message_print}\n" + "*"*100)
    logger.info(f"*** Full Send Message ***\n" + send_message)
    print(my_msg_user)

    s_time = time.time() # 推論開始時間

    try_num = 0

    # モデルが読み込まれていなかったらこの時点で待機
    if not self.proc_load_model(actor, stop_event):
      return None

    while True:
      try_num += 1
      # 途中終了
      if stop_event.is_set(): return None
      if self.check_msg_change(chk_message): return ""
      
      sys_prompt_t = sys_prompt

      # 最大返信トークン数
      max_new_tokens = actor.max_new_tokens

      # 途中終了
      if stop_event.is_set(): return None
      if self.check_msg_change(chk_message): return ""

      # チャットクラスからレスポンス取得
      response = self.chat.get_response_actor(actor, send_message, sys_prompt_t, input_img_set, input_vdo_set, max_new_tokens, actor.temperature, temp_type, user_name, cfg.no_chat_key, sent_num, False, no_print, name_no_msg_flg)

      if not response or response == "":
        # レスポンスが取得できなかったら再取得
        continue
      else:
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

    self.last_message = f"{actor_name}: {res_simple}"

    # 履歴に追加(+チャット欄更新なし) ※チャットランを更新しないのは文字を段階的に表示するため
    self.add_txt_chat(actor_name, res_simple, res_time=time_text, eff_texts=res_eff_texts, chat_update=False, new_line_num=new_line, font_tag=actor.chat_name)

    view_sec = cfg.chat_view_sec if hasattr(cfg, "chat_view_sec") else 0.1
    if hasattr(actor, "chat_view_sec"):
      view_sec = actor.chat_view_sec
      
    # チャット欄にレスポンスの効果を書き込む
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

    # レスポンスを画面に書き込む
    self.write_chat_actor(actor.chat_name, chat_response, time_text, new_line, view_sec=view_sec, stop_event=stop_event)

    return f"{actor_name}: {chat_response}"

  # 送信メッセージが変わっているかチェック
  def check_msg_change(self, message):
    if not message:
      return False
    elif self.last_message != message:
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
    self.chat.history_rewind(rew_num)
    self.update_chat()
    self.bef_message = ""

  # レスポンスを書き込む
  def write_chat_actor(self, chat_name:str, response, res_time=None, new_line=0, view_sec=0, stop_event=threading.Event()):
    if self.txt_chat.get("1.0", tk.END).strip() != "":
      new_line += 1 # 空白でないときは先頭に改行

    while self.is_write_chat:
      time.sleep(0.1)

    self.is_write_chat = True
    self.write_chat_text(f"{chat_name}: ", new_line, font_tag=chat_name, req_see_flg=True)

    if view_sec == 0:
      # 一度に全ての文を出力
      self.write_chat_text(response, font_tag=chat_name, req_see_flg=True)
    else:
      i = 0
      while i < len(response):
        # self.txt_chat.insert("end", add_text)
        self.write_chat_text(response[i], font_tag=chat_name, req_see_flg=True)
        time.sleep(view_sec)
        i += 1

    self.is_write_chat = False

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

  # メッセージ送信時のコントロールの活性・非活性
  def send_menu_enabled(self, enable_flg=True, train_flg=False):
    if enable_flg:
      tk_state = tk.NORMAL
      text=BUTTON_SEND
    else:
      tk_state = tk.DISABLED
      text=BUTTON_SEND_STOP

    if train_flg:
      # 学習時は[送信]のまま非活性
      self.btn_send.config(state=tk_state)
    else:
      # ボタンの[送信][中断]切り替え
      self.btn_send.config(text=text)

    self.menu_file.entryconfig(MENU_CHAT_CLEAR, state=tk_state)
    self.menu_file.entryconfig(MENU_ACTOR_READ, state=tk_state)
    self.menu_file.entryconfig(MENU_ACTOR_RELEASE, state=tk_state)
    self.menubar.entryconfig(MENU_EDIT, state=tk_state)
    
  # モデルが有効かどうかでコントロールの活性・非活性
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
      # 履歴からテキスト作成
      text = self.get_dic_to_text(dic)
      if chat_text != "":
        chat_text += "\n"
      chat_text += text
  
    return chat_text
  
  # チャット画面を更新
  def update_chat(self):
    self.txt_chat.config(state=tk.NORMAL)
    self.txt_chat.delete("1.0", "end")
    
    # 履歴
    chat_value = self.history_list_to_text()
    
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
    
    if "  " in text_eff:
      print(text_eff)
      text_eff.replace("  ", " ")

    self.chat.history_dic_list.append({"name": name, "text": text, "effects": eff_texts, "response": res_time})
    log_text_eff_time = text_eff_time.replace("\n","")

    print("*** Message/Response with time " + "*"*69 + f"\n* {log_text_eff_time}\n" + "*"*100)
    logger.info("*** Message/Response with time ***\n" + log_text_eff_time)

    if self.view_chat == 0:
      add_text = name_text
    elif self.view_chat == 1:
      add_text = text_eff

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

  # 開始時のモデルのロード
  def start_load_actor(self, actor_path):
    self.actors = [None] # 一人分の枠を確保
    ret = False

    if hasattr(self.cfg, "actor_auto_load") and self.cfg.actor_auto_load:
      ret = self.load_actor(actor_path, c_id=0, init_flg=True)
    self.actor_menu_enable(ret)

  # モデルのロード
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

      # モデル画像をセット
      self.actors[c_id].load_main_img()

      # メインモデルのみ変更する項目
      if c_id == 0:
        # プロフィールをセット
        self.txt_prof.config(state=tk.NORMAL)
        self.txt_prof.delete("1.0", tk.END)
        self.txt_prof.insert("1.0", self.actors[c_id].get_profile())
        self.txt_prof.config(state=tk.DISABLED)

      if self.actors[c_id].model_enable:
        if self.cfg.thread_model_load and not info_only:
          # モデルを別スレッドでロード
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

    return image

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
  def get_input_image(self, img_max_size=-1):

    input_images = []
    input_img_paths = []

    if self.win_picture:
      if len(self.win_picture.image_dic_list) > 0:
        
        for dic in self.win_picture.image_dic_list:
          image = dic["data"]
          path = dic["path"]
         
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
