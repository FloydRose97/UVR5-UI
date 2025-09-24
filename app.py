import os
import sys
import subprocess
import re
import platform
import torch
import logging
import yt_dlp
import json
import copy
import tempfile
import shutil
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import gradio as gr
import urllib.parse
import assets.themes.loadThemes as loadThemes
from audio_separator.separator import Separator
from audio_separator.separator.uvr_lib_v5 import spec_utils
from assets.i18n.i18n import I18nAuto
from argparse import ArgumentParser
from assets.presence.discord_presence import RPCManager, track_presence

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)
config_file = os.path.join(now_dir, "assets", "config.json")
models_file = os.path.join(now_dir, "assets", "models.json")
default_settings_file = os.path.join(now_dir, "assets", "default_settings.json")
custom_settings_file = os.path.join(now_dir, "assets", "custom_settings.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

if os.path.isdir("env"):
    if platform.system() == "Windows":
        python_location = ".\\env\\python.exe"
        separator_location = ".\\env\\Scripts\\audio-separator.exe"
    elif platform.system() == "Linux":
        python_location = "env/bin/python"
        separator_location = "env/bin/audio-separator"
else:
    python_location = None
    separator_location = "audio-separator"

if __name__ == "__main__":
    parser = ArgumentParser(
       description="Separate audio into multiple stems",
       epilog="Example: python app.py --share --listen-port 8080 --open"
    )
    parser.add_argument(
       "--share",
       action="store_true",
       help="Enable sharing of the interface through Gradio's temporary URLs"
    )
    parser.add_argument(
       "--listen-port",
       type=int,
       default=9999,
       help="The listening port that the server will use (default: 9999)"
    )
    parser.add_argument(
       "--open",
       action="store_true",
       help="Automatically open the interface in the default web browser"
    )   
    args = parser.parse_args()

#=========================#
#     Roformer Models     #
#=========================#
roformer_models = {
    'BS-Roformer-Viperx-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'BS-Roformer-Viperx-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    'BS-Roformer-Viperx-1053': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
    'Mel-Roformer-Viperx-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    'BS-Roformer-De-Reverb': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
    'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
    'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr' : 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'MelBand Roformer | Denoise-Debleed by Gabox' : 'mel_band_roformer_denoise_debleed_gabox.ckpt',
    'Mel-Roformer-Karaoke-Aufr33-Viperx': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    'MelBand Roformer | Karaoke by Gabox' : 'mel_band_roformer_karaoke_gabox.ckpt',
    'MelBand Roformer | Karaoke V2 by Gabox' : 'mel_band_roformer_karaoke_gabox_v2.ckpt',
    'MelBand Roformer | Karaoke by becruily' : 'mel_band_roformer_karaoke_becruily.ckpt',
    'MelBand Roformer | Vocals by Kimberley Jensen' : 'vocals_mel_band_roformer.ckpt',
    'MelBand Roformer Kim | FT by unwa' : 'mel_band_roformer_kim_ft_unwa.ckpt',
    'MelBand Roformer Kim | FT 2 by unwa' : 'mel_band_roformer_kim_ft2_unwa.ckpt',
    'MelBand Roformer Kim | FT 2 Bleedless by unwa' : 'mel_band_roformer_kim_ft2_bleedless_unwa.ckpt',
    'MelBand Roformer Kim | FT 3 by unwa' : 'mel_band_roformer_kim_ft3_unwa.ckpt',
    'MelBand Roformer Kim | Inst V1 by Unwa' : 'melband_roformer_inst_v1.ckpt',
    'MelBand Roformer Kim | Inst V1 Plus by Unwa' : 'melband_roformer_inst_v1_plus.ckpt',
    'MelBand Roformer Kim | Inst V1 (E) by Unwa' : 'melband_roformer_inst_v1e.ckpt',
    'MelBand Roformer Kim | Inst V1 (E) Plus by Unwa' : 'melband_roformer_inst_v1e_plus.ckpt',
    'MelBand Roformer Kim | Inst V2 by Unwa' : 'melband_roformer_inst_v2.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V1 by Unwa' : 'melband_roformer_instvoc_duality_v1.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V2 by Unwa' : 'melband_roformer_instvox_duality_v2.ckpt',
    'MelBand Roformer | Vocals by becruily' : 'mel_band_roformer_vocals_becruily.ckpt',
    'MelBand Roformer | Instrumental by becruily' : 'mel_band_roformer_instrumental_becruily.ckpt',
    'MelBand Roformer | Vocals Fullness by Aname' : 'mel_band_roformer_vocal_fullness_aname.ckpt',
    'BS Roformer | Vocals by Gabox' : 'bs_roformer_vocals_gabox.ckpt',
    'MelBand Roformer | Vocals by Gabox' : 'mel_band_roformer_vocals_gabox.ckpt',
    'MelBand Roformer | Vocals V2 by Gabox' : 'mel_band_roformer_vocals_v2_gabox.ckpt',
    'MelBand Roformer | Vocals FV1 by Gabox' : 'mel_band_roformer_vocals_fv1_gabox.ckpt',
    'MelBand Roformer | Vocals FV2 by Gabox' : 'mel_band_roformer_vocals_fv2_gabox.ckpt',
    'MelBand Roformer | Vocals FV3 by Gabox' : 'mel_band_roformer_vocals_fv3_gabox.ckpt',
    'MelBand Roformer | Vocals FV4 by Gabox' : 'mel_band_roformer_vocals_fv4_gabox.ckpt',
    'MelBand Roformer | Vocals FV5 by Gabox' : 'mel_band_roformer_vocals_fv5_gabox.ckpt',
    'MelBand Roformer | Vocals FV6 by Gabox' : 'mel_band_roformer_vocals_fv6_gabox.ckpt',
    'MelBand Roformer | Instrumental by Gabox' : 'mel_band_roformer_instrumental_gabox.ckpt',
    'MelBand Roformer | Instrumental 2 by Gabox' : 'mel_band_roformer_instrumental_2_gabox.ckpt',
    'MelBand Roformer | Instrumental 3 by Gabox' : 'mel_band_roformer_instrumental_3_gabox.ckpt',
    'MelBand Roformer | Instrumental Bleedless V1 by Gabox' : 'mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt',
    'MelBand Roformer | Instrumental Bleedless V2 by Gabox' : 'mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt',
    'MelBand Roformer | Instrumental Bleedless V3 by Gabox' : 'mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness V1 by Gabox' : 'mel_band_roformer_instrumental_fullness_v1_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness V2 by Gabox' : 'mel_band_roformer_instrumental_fullness_v2_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness V3 by Gabox' : 'mel_band_roformer_instrumental_fullness_v3_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness Noisy V4 by Gabox' : 'mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt',
    'MelBand Roformer | INSTV5 by Gabox' : 'mel_band_roformer_instrumental_instv5_gabox.ckpt',
    'MelBand Roformer | INSTV5N by Gabox' : 'mel_band_roformer_instrumental_instv5n_gabox.ckpt',
    'MelBand Roformer | INSTV6 by Gabox' : 'mel_band_roformer_instrumental_instv6_gabox.ckpt',
    'MelBand Roformer | INSTV6N by Gabox' : 'mel_band_roformer_instrumental_instv6n_gabox.ckpt',
    'MelBand Roformer | INSTV7 by Gabox' : 'mel_band_roformer_instrumental_instv7_gabox.ckpt',
    'MelBand Roformer | INSTV7N by Gabox' : 'mel_band_roformer_instrumental_instv7n_gabox.ckpt',
    'MelBand Roformer | INSTV8 by Gabox' : 'mel_band_roformer_instrumental_instv8_gabox.ckpt',
    'MelBand Roformer | INSTV8N by Gabox' : 'mel_band_roformer_instrumental_instv8n_gabox.ckpt',
    'MelBand Roformer | Instrumental FV7z by Gabox' : 'mel_band_roformer_instrumental_fv7z_gabox.ckpt',
    'MelBand Roformer | Instrumental FV8 by Gabox' : 'mel_band_roformer_instrumental_fv8_gabox.ckpt',
    'MelBand Roformer | Instrumental FVX by Gabox' : 'mel_band_roformer_instrumental_fvx_gabox.ckpt',
    'MelBand Roformer | De-Reverb by anvuew' : 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
    'MelBand Roformer | De-Reverb Less Aggressive by anvuew' : 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
    'MelBand Roformer | De-Reverb Mono by anvuew' : 'dereverb_mel_band_roformer_mono_anvuew.ckpt',
    'MelBand Roformer | De-Reverb Big by Sucial' : 'dereverb_big_mbr_ep_362.ckpt',
    'MelBand Roformer | De-Reverb Super Big by Sucial' : 'dereverb_super_big_mbr_ep_346.ckpt',
    'MelBand Roformer | De-Reverb-Echo by Sucial' : 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
    'MelBand Roformer | De-Reverb-Echo V2 by Sucial' : 'dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt',
    'MelBand Roformer | De-Reverb-Echo Fused by Sucial' : 'dereverb_echo_mbr_fused.ckpt',
    'MelBand Roformer Kim | SYHFT by SYH99999' : 'MelBandRoformerSYHFT.ckpt',
    'MelBand Roformer Kim | SYHFT V2 by SYH99999' : 'MelBandRoformerSYHFTV2.ckpt',
    'MelBand Roformer Kim | SYHFT V2.5 by SYH99999' : 'MelBandRoformerSYHFTV2.5.ckpt',
    'MelBand Roformer Kim | SYHFT V3 by SYH99999' : 'MelBandRoformerSYHFTV3Epsilon.ckpt',
    'MelBand Roformer Kim | Big SYHFT V1 by SYH99999' : 'MelBandRoformerBigSYHFTV1.ckpt',
    'MelBand Roformer Kim | Big Beta 4 FT by unwa' : 'melband_roformer_big_beta4.ckpt',
    'MelBand Roformer Kim | Big Beta 5e FT by unwa' : 'melband_roformer_big_beta5e.ckpt',
    'MelBand Roformer | Big Beta 6 by unwa' : 'melband_roformer_big_beta6.ckpt',
    'MelBand Roformer | Big Beta 6X by unwa' : 'melband_roformer_big_beta6x.ckpt',
    'BS Roformer | Vocals Revive by Unwa' : 'bs_roformer_vocals_revive_unwa.ckpt',
    'BS Roformer | Vocals Revive V2 by Unwa' : 'bs_roformer_vocals_revive_v2_unwa.ckpt',
    'BS Roformer | Vocals Revive V3e by Unwa' : 'bs_roformer_vocals_revive_v3e_unwa.ckpt',
    'BS Roformer | Chorus Male-Female by Sucial' : 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',
    'BS Roformer | Male-Female by aufr33' : 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
    'MelBand Roformer | Aspiration by Sucial' : 'aspiration_mel_band_roformer_sdr_18.9845.ckpt',
    'MelBand Roformer | Aspiration Less Aggressive by Sucial' : 'aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt',
    'MelBand Roformer | Bleed Suppressor V1 by unwa-97chris' : 'mel_band_roformer_bleed_suppressor_v1.ckpt',
    'BS Roformer | Vocals Resurrection by unwa' : 'bs_roformer_vocals_resurrection_unwa.ckpt',
    'BS Roformer | Instrumental Resurrection by unwa' : 'bs_roformer_instrumental_resurrection_unwa.ckpt'
}

#=========================#
#      MDX23C Models      #
#=========================#
mdx23c_models = [
    'MDX23C_D1581.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
    'MDX23C-De-Reverb-aufr33-jarredou.ckpt',
    'MDX23C-DrumSep-aufr33-jarredou.ckpt'
]

#=========================#
#     MDXN-NET Models     #
#=========================#
mdxnet_models = [
    'UVR-MDX-NET-Inst_full_292.onnx',
    'UVR-MDX-NET_Inst_187_beta.onnx',
    'UVR-MDX-NET_Inst_82_beta.onnx',
    'UVR-MDX-NET_Inst_90_beta.onnx',
    'UVR-MDX-NET_Main_340.onnx',
    'UVR-MDX-NET_Main_390.onnx',
    'UVR-MDX-NET_Main_406.onnx',
    'UVR-MDX-NET_Main_427.onnx',
    'UVR-MDX-NET_Main_438.onnx',
    'UVR-MDX-NET-Inst_HQ_1.onnx',
    'UVR-MDX-NET-Inst_HQ_2.onnx',
    'UVR-MDX-NET-Inst_HQ_3.onnx',
    'UVR-MDX-NET-Inst_HQ_4.onnx',
    'UVR-MDX-NET-Inst_HQ_5.onnx',
    'UVR_MDXNET_Main.onnx',
    'UVR-MDX-NET-Inst_Main.onnx',
    'UVR_MDXNET_1_9703.onnx',
    'UVR_MDXNET_2_9682.onnx',
    'UVR_MDXNET_3_9662.onnx',
    'UVR-MDX-NET-Inst_1.onnx',
    'UVR-MDX-NET-Inst_2.onnx',
    'UVR-MDX-NET-Inst_3.onnx',
    'UVR_MDXNET_KARA.onnx',
    'UVR_MDXNET_KARA_2.onnx',
    'UVR_MDXNET_9482.onnx',
    'UVR-MDX-NET-Voc_FT.onnx',
    'Kim_Vocal_1.onnx',
    'Kim_Vocal_2.onnx',
    'Kim_Inst.onnx',
    'Reverb_HQ_By_FoxJoy.onnx',
    'UVR-MDX-NET_Crowd_HQ_1.onnx',
    'kuielab_a_vocals.onnx',
    'kuielab_a_other.onnx',
    'kuielab_a_bass.onnx',
    'kuielab_a_drums.onnx',
    'kuielab_b_vocals.onnx',
    'kuielab_b_other.onnx',
    'kuielab_b_bass.onnx',
    'kuielab_b_drums.onnx',
]

#========================#
#     VR-ARCH Models     #
#========================#
vrarch_models = [
    '1_HP-UVR.pth',
    '2_HP-UVR.pth',
    '3_HP-Vocal-UVR.pth',
    '4_HP-Vocal-UVR.pth',
    '5_HP-Karaoke-UVR.pth',
    '6_HP-Karaoke-UVR.pth',
    '7_HP2-UVR.pth',
    '8_HP2-UVR.pth',
    '9_HP2-UVR.pth',
    '10_SP-UVR-2B-32000-1.pth',
    '11_SP-UVR-2B-32000-2.pth',
    '12_SP-UVR-3B-44100.pth',
    '13_SP-UVR-4B-44100-1.pth',
    '14_SP-UVR-4B-44100-2.pth',
    '15_SP-UVR-MID-44100-1.pth',
    '16_SP-UVR-MID-44100-2.pth',
    '17_HP-Wind_Inst-UVR.pth',
    'UVR-De-Echo-Aggressive.pth',
    'UVR-De-Echo-Normal.pth',
    'UVR-DeEcho-DeReverb.pth',
    'UVR-De-Reverb-aufr33-jarredou.pth',
    'UVR-DeNoise-Lite.pth',
    'UVR-DeNoise.pth',
    'UVR-BVE-4B_SN-44100-1.pth',
    'UVR-BVE-4B_SN-44100-2.pth',
    'MGM_HIGHEND_v4.pth',
    'MGM_LOWEND_A_v4.pth',
    'MGM_LOWEND_B_v4.pth',
    'MGM_MAIN_v4.pth',
]

#=======================#
#     DEMUCS Models     #
#=======================#
demucs_models = [
    'htdemucs_ft.yaml',
    'htdemucs_6s.yaml',
    'htdemucs.yaml',
    'hdemucs_mmi.yaml',
]

ensemble_model_choices = []
ensemble_model_map = {}


def register_ensemble_model(type_key, type_label, display_name, filename):
    option_label = f"{type_label} | {display_name}"
    ensemble_model_choices.append(option_label)
    ensemble_model_map[option_label] = {
        "type_key": type_key,
        "type_label": type_label,
        "display": display_name,
        "filename": filename,
    }


for display_name, file_name in roformer_models.items():
    register_ensemble_model("roformer", "BS/Mel Roformer", display_name, file_name)

for file_name in mdx23c_models:
    register_ensemble_model("mdx23c", "MDX23C", file_name, file_name)

for file_name in mdxnet_models:
    register_ensemble_model("mdxnet", "MDX-NET", file_name, file_name)

for file_name in vrarch_models:
    register_ensemble_model("vrarch", "VR Arch", file_name, file_name)

for file_name in demucs_models:
    register_ensemble_model("demucs", "Demucs", file_name, file_name)

output_format = [
    'wav',
    'flac',
    'mp3',
    'ogg',
    'opus',
    'm4a',
    'aiff',
    'ac3'
]

found_files = []
logs = []


class StatusReporter:
    """Helper to mirror progress updates to both the UI and terminal."""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.messages = []

    def emit(self, message, progress_value=None):
        if message:
            print(message, flush=True)
            self.messages.append(message)

        if self.progress_callback is not None and progress_value is not None:
            self.progress_callback(progress_value, desc=message)

        return "\n".join(self.messages)
out_dir = "./outputs"
models_dir = "./models"
extensions = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")

def load_config_presence():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        return config["discord_presence"]

def initialize_presence():
    if load_config_presence():
        RPCManager.start_presence()

initialize_presence()

def download_audio(url, output_dir="ytdl"):

    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '32',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessor_args': [
            '-acodec', 'pcm_f32le'
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info['title']

            ydl.download([url])

            file_path = os.path.join(output_dir, f"{video_title}.wav")

            if os.path.exists(file_path):
                return os.path.abspath(file_path)
            else:
                raise Exception("Something went wrong")

    except Exception as e:
        raise Exception(f"Error extracting audio with yt-dlp: {str(e)}")

def leaderboard(list_filter):
    try:
        if python_location:
            command = [python_location, separator_location, "-l", f"--list_filter={list_filter}"]
        else:
            command = [separator_location, "-l", f"--list_filter={list_filter}"]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return "<table border='1'>" + "".join(
            f"<tr style='{'font-weight: bold; font-size: 1.2em;' if i == 0 else ''}'>" +
            "".join(f"<td>{cell}</td>" for cell in re.split(r"\s{2,}", line.strip())) +
            "</tr>"
            for i, line in enumerate(re.findall(r"^(?!-+)(.+)$", result.stdout.strip(), re.MULTILINE))
        ) + "</table>"

    except Exception as e:
        return f"Error: {e}"
    
def get_language_settings():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)

    if config["lang"]["override"] == False:
        return "Language automatically detected by system"
    else:
        return config["lang"]["selected_lang"]
    
def save_lang_settings(selected_language):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)

    if selected_language == "Language automatically detected by system":
        config["lang"]["override"] = False
    else:
        config["lang"]["override"] = True
        config["lang"]["selected_lang"] = selected_language

    gr.Info(i18n("Language have been saved. Restart UVR5 UI to apply the changes"))

    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)

def alternative_model_downloader(method, key, output_dir="models", progress=gr.Progress()):
    logs.clear()

    with open(models_file, 'r', encoding='utf-8') as file:
        model_data = json.load(file)
    
    if key not in model_data:
        return f"Model '{key}' cannot be found."
    
    total_files = len(model_data[key])
    progress(0, desc="Starting downloads...")

    def record(message):
        print(message, flush=True)
        logs.append(message)

    for i, url in enumerate(model_data[key]):
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        full_name = os.path.join(output_dir, filename)

        if os.path.exists(full_name):
            record(f"{filename} already exists.")
            continue

        progress((i + 0.1) / total_files, desc=f"Starting download of {filename} ({i+1}/{total_files})")

        if method == 'wget':
            cmd = ['wget', '--progress=bar:force', '-O', full_name, url]
        elif method == 'curl':
            cmd = ['curl', '-L', '-#', '-o', full_name, url]

        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stderr:
                if method == 'wget' and '%' in line:
                    try:
                        percent = int(line.strip().split('%')[0].split()[-1])
                        file_progress = percent / 100.0
                        total_progress = (i + file_progress) / total_files
                        progress(total_progress, desc=f"File {i+1}/{total_files}: {filename} ({percent}%)")
                    except (ValueError, IndexError):
                        pass
                elif method == 'curl' and '##' in line:
                    try:
                        hash_count = line.count('#')
                        file_progress = min(hash_count / 50.0, 1.0)
                        total_progress = (i + file_progress) / total_files
                        percent = int(file_progress * 100)
                        progress(total_progress, desc=f"File {i+1}/{total_files}: {filename} ({percent}%)")
                    except Exception:
                        pass
            
            process.wait()
            if process.returncode != 0:
                record(f"Error downloading {filename}")
            else:
                record(f"{filename} downloaded successfully!")
                progress((i + 1) / total_files, desc=f"File {i+1}/{total_files} completed")

        except Exception as e:
            record(f"Error running download command: {str(e)}")

    progress(1.0, desc="Download process completed")
    return "\n".join(logs)

def read_main_config():
    try:
        with open(config_file, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading main config file '{config_file}': {e}")
        gr.Warning(i18n("Error reading main config file"))
    
def write_main_config(data):
    try:
        with open(config_file, "w", encoding="utf8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error writing to main config file '{config_file}': {e}")
        gr.Warning(i18n("Error writing to main config file"))

def load_settings_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading settings file '{filepath}': {e}")
        gr.Warning(i18n("Error reading settings file"))
        return None
    
def get_initial_settings():
    main_config = read_main_config()
    load_custom = main_config.get('load_custom_settings', False)

    settings_to_load = {}
    default_settings = load_settings_from_file(default_settings_file)

    if load_custom:
        print("Attempting to load custom settings...")
        custom_settings = load_settings_from_file(custom_settings_file)
        if custom_settings:
            settings_to_load = copy.deepcopy(default_settings)
            for section, params in custom_settings.items():
                if section in settings_to_load:
                    for key, value in params.items():
                        settings_to_load[section][key] = value
                else:
                    settings_to_load[section] = params
            print("Custom settings loaded successfully.")
        else:
            print("Custom settings file not found or invalid. Falling back to default settings.")
            settings_to_load = default_settings
    else:
        print("Loading default settings...")
        settings_to_load = default_settings

    return settings_to_load

initial_settings = get_initial_settings()

def get_all_components(components_dict):
    all_comps = []
    for section in components_dict.values():
        all_comps.extend(section.values())
    return all_comps

def save_current_settings(*values):
    global components
    try:
        current_config_data = {}
        value_index = 0
        for section_name, section_comps in components.items():
            current_config_data[section_name] = {}
            for comp_name in section_comps.keys():
                current_config_data[section_name][comp_name] = values[value_index]
                value_index += 1

        with open(custom_settings_file, 'w', encoding='utf-8') as f:
            json.dump(current_config_data, f, indent=4)

        main_config = read_main_config()
        main_config['load_custom_settings'] = True
        write_main_config(main_config)
        gr.Info(i18n("Current settings saved successfully! They will be loaded next time"))
    except Exception as e:
        print(f"Error saving settings: {e}")
        gr.Warning(i18n("Error saving settings"))

def reset_settings_to_default():
    global components, default_settings_file
    updates = []
    all_comps_flat = get_all_components(components)
    try:
        default_settings = load_settings_from_file(default_settings_file)
        for section_name, section_comps in components.items():
            for comp_name, comp_instance in section_comps.items():
                default_value = default_settings.get(section_name, {}).get(comp_name, None)

                if isinstance(comp_instance, gr.Dropdown) and hasattr(comp_instance, 'choices') and default_value is not None:
                    if default_value not in comp_instance.choices:
                        print(f"Warning: Default value '{default_value}' for '{comp_name}' not in choices {comp_instance.choices}. Setting to None.")
                        default_value = None

                updates.append(gr.update(value=default_value))

        main_config = read_main_config()
        main_config['load_custom_settings'] = False
        write_main_config(main_config)

        gr.Info(i18n("Settings reset to default. Default settings will be loaded next time"))
        return updates

    except Exception as e:
        print(f"Error resetting settings: {e}")
        gr.Warning(i18n("Error resetting settings"))
        return [gr.update() for _ in all_comps_flat]

ensemble_stem_pattern = re.compile(r"^(?P<audio>.+?)_\((?P<stem>.+?)\)_(?P<model>.+)$")


def build_ensemble_stem_map(file_paths):
    stem_map = {}
    for index, file_path in enumerate(file_paths, start=1):
        filename = os.path.basename(file_path)
        stem_name, _ = os.path.splitext(filename)
        match = ensemble_stem_pattern.match(stem_name)
        if match:
            audio_base = match.group("audio").strip()
            stem_token = match.group("stem").strip()
        else:
            audio_base = stem_name.strip()
            stem_token = stem_name.strip()

        normalized_key = stem_token.lower()
        if normalized_key in stem_map:
            raise ValueError(f"Duplicate stem detected: {stem_token}")

        display_name = stem_token.replace("_", " ").strip() or f"Stem {index}"
        stem_map[normalized_key] = {
            "path": file_path,
            "stem_token": stem_token or display_name,
            "stem_display": display_name,
            "audio_base_token": audio_base or stem_name,
        }

    return stem_map


def combine_ensemble_stems(model_stem_maps):
    if not model_stem_maps:
        raise ValueError("No stems were provided for ensembling")

    base_keys = list(model_stem_maps[0].keys())
    base_key_set = set(base_keys)

    for stem_map in model_stem_maps[1:]:
        other_keys = set(stem_map.keys())
        if other_keys != base_key_set:
            difference = ", ".join(sorted(base_key_set.symmetric_difference(other_keys)))
            raise ValueError(f"Incompatible stems across models: {difference}")

    combined_outputs = []

    for stem_key in base_keys:
        reference_entry = model_stem_maps[0][stem_key]
        sample_rate = None
        max_length = 0
        max_channels = 0
        collected_arrays = []

        for stem_map in model_stem_maps:
            audio_path = stem_map[stem_key]["path"]
            audio_array, sr = sf.read(audio_path, always_2d=True)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError("Sample rate mismatch between ensemble models")

            max_length = max(max_length, audio_array.shape[0])
            max_channels = max(max_channels, audio_array.shape[1])
            collected_arrays.append(audio_array)

        aligned_arrays = []
        for array in collected_arrays:
            if array.shape[1] < max_channels:
                if array.shape[1] == 1 and max_channels == 2:
                    array = np.repeat(array, 2, axis=1)
                else:
                    array = np.pad(array, ((0, 0), (0, max_channels - array.shape[1])), mode='constant')
            elif array.shape[1] > max_channels:
                array = array[:, :max_channels]

            if array.shape[0] < max_length:
                array = np.pad(array, ((0, max_length - array.shape[0]), (0, 0)), mode='constant')

            aligned_arrays.append(array.astype(np.float64))

        stacked = np.stack(aligned_arrays, axis=0)
        averaged = np.mean(stacked, axis=0)

        combined_outputs.append({
            "stem_token": reference_entry["stem_token"],
            "stem_display": reference_entry["stem_display"],
            "audio_base_token": reference_entry["audio_base_token"],
            "data": averaged.astype(np.float32),
            "sample_rate": sample_rate,
        })

    return combined_outputs


def write_ensemble_outputs(combined_outputs, output_format, normalization_threshold, amplification_threshold):
    ensemble_results = []
    os.makedirs(out_dir, exist_ok=True)
    export_format = (output_format or "wav").lower()

    for entry in combined_outputs:
        file_name = f"{entry['audio_base_token']}_({entry['stem_token']})_Ensemble.{export_format}"
        destination = os.path.join(out_dir, file_name)

        audio_data = entry["data"]
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]

        peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
        if peak >= 1e-9:
            normalized = spec_utils.normalize(audio_data.copy(), max_peak=normalization_threshold, min_peak=amplification_threshold)
        else:
            normalized = np.zeros_like(audio_data)

        normalized = np.clip(normalized, -1.0, 1.0)
        int_data = (normalized * 32767).astype(np.int16)

        if int_data.ndim == 1:
            int_data = int_data[:, np.newaxis]

        channels = int_data.shape[1]
        if channels > 1:
            interleaved = np.empty((channels * int_data.shape[0],), dtype=np.int16)
            for channel_index in range(channels):
                interleaved[channel_index::channels] = int_data[:, channel_index]
        else:
            interleaved = int_data[:, 0]

        audio_segment = AudioSegment(
            interleaved.tobytes(),
            frame_rate=entry["sample_rate"],
            sample_width=int_data.dtype.itemsize,
            channels=channels,
        )

        export_format_name = export_format
        if export_format_name == "m4a":
            export_format_name = "mp4"
        elif export_format_name == "mka":
            export_format_name = "matroska"

        bitrate = "320k" if export_format == "mp3" else None
        audio_segment.export(destination, format=export_format_name, bitrate=bitrate)
        ensemble_results.append((entry["stem_display"], destination))

    return ensemble_results


def run_model_for_ensemble(audio_path, model_info, single_stem, normalization_threshold, amplification_threshold,
                           roformer_params, mdx23c_params, mdxnet_params, vrarch_params, demucs_params):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="ensemble_tmp_", dir=out_dir)
    success = False

    try:
        separator_kwargs = {
            "log_level": logging.WARNING,
            "model_file_dir": models_dir,
            "output_dir": temp_dir,
            "output_format": "wav",
            "use_autocast": use_autocast,
            "normalization_threshold": normalization_threshold,
            "amplification_threshold": amplification_threshold,
            "output_single_stem": single_stem,
        }

        model_type = model_info["type_key"]

        if model_type in {"roformer", "mdx23c"}:
            params = roformer_params if model_type == "roformer" else mdx23c_params
            separator_kwargs["mdxc_params"] = {
                "segment_size": params["segment_size"],
                "override_model_segment_size": params["override_segment_size"],
                "batch_size": params["batch_size"],
                "overlap": params["overlap"],
            }
        elif model_type == "mdxnet":
            separator_kwargs["mdx_params"] = {
                "hop_length": mdxnet_params["hop_length"],
                "segment_size": mdxnet_params["segment_size"],
                "overlap": mdxnet_params["overlap"],
                "batch_size": mdxnet_params["batch_size"],
                "enable_denoise": mdxnet_params["denoise"],
            }
        elif model_type == "vrarch":
            separator_kwargs["vr_params"] = {
                "batch_size": vrarch_params["batch_size"],
                "window_size": vrarch_params["window_size"],
                "aggression": vrarch_params["aggression"],
                "enable_tta": vrarch_params["tta"],
                "enable_post_process": vrarch_params["post_process"],
                "post_process_threshold": vrarch_params["post_process_threshold"],
                "high_end_process": vrarch_params["high_end_process"],
            }
        elif model_type == "demucs":
            separator_kwargs["demucs_params"] = {
                "batch_size": demucs_params["batch_size"],
                "segment_size": demucs_params["segment_size"],
                "shifts": demucs_params["shifts"],
                "overlap": demucs_params["overlap"],
                "segments_enabled": demucs_params["segments_enabled"],
            }
        else:
            raise ValueError(f"Unsupported ensemble model type: {model_type}")

        separator = Separator(**separator_kwargs)
        model_filename = model_info["filename"]
        model_path = os.path.join(models_dir, model_filename)

        if not os.path.exists(model_path):
            gr.Info(i18n("This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.").format(model=model_info["display"]))

        try:
            separator.load_model(model_filename=model_filename)
        except SystemExit as exc:
            message = i18n(
                "{model} exited while loading. The model file may be corrupt. Delete {path} and try again."
            ).format(model=model_info["display"], path=model_path)
            raise gr.Error(message) from exc
        except Exception as exc:
            raise gr.Error(
                i18n("Failed to load {model}: {error}").format(model=model_info["display"], error=str(exc))
            ) from exc

        try:
            separation = separator.separate(audio_path)
        except SystemExit as exc:
            message = i18n(
                "{model} exited unexpectedly during separation. Check the model file and try again."
            ).format(model=model_info["display"])
            raise gr.Error(message) from exc

        absolute_paths = [os.path.join(temp_dir, file_name) for file_name in separation]

        stem_map = build_ensemble_stem_map(absolute_paths)
        if not stem_map:
            raise RuntimeError(i18n("Model {model} did not produce any stems.").format(model=model_info["display"]))

        success = True
        return {"temp_dir": temp_dir, "stems": stem_map}

    except gr.Error:
        raise
    except Exception as exc:
        raise RuntimeError(
            i18n("Model {model} failed: {error}").format(model=model_info["display"], error=str(exc))
        ) from exc
    finally:
        if not success:
            shutil.rmtree(temp_dir, ignore_errors=True)

components = {
    "Roformer": {}, "MDX23C": {}, "MDX-NET": {}, "VR Arch": {}, "Demucs": {}, "Ensemble": {}
}

@track_presence("Performing BS/Mel Roformer Separation")
def roformer_separator(audio, model_key, out_format, segment_size, override_seg_size, overlap, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress(track_tqdm=True)):
    roformer_model = roformer_models[model_key]
    model_path = os.path.join(models_dir, roformer_model)
    reporter = StatusReporter(progress)

    def empty_audio_updates():
        return (gr.update(value=None), gr.update(value=None))

    try:
        status_text = reporter.emit(i18n("Preparing separation..."), 0.0)
        yield (status_text, *empty_audio_updates())

        if not os.path.exists(model_path):
            download_message = i18n("Downloading model: {model}...").format(model=model_key)
            status_text = reporter.emit(download_message)
            yield (status_text, *empty_audio_updates())
            gr.Info(i18n("This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.").format(model=model_key))

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=models_dir,
            output_dir=out_dir,
            output_format=out_format,
            use_autocast=use_autocast,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            output_single_stem=single_stem,
            mdxc_params={
                "segment_size": segment_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
            }
        )

        status_text = reporter.emit(i18n("Loading model..."), 0.2)
        yield (status_text, *empty_audio_updates())
        separator.load_model(model_filename=roformer_model)

        status_text = reporter.emit(i18n("Separating audio..."), 0.7)
        yield (status_text, *empty_audio_updates())
        separation = separator.separate(audio)

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        status_text = reporter.emit(i18n("Finalizing outputs..."), 0.9)
        yield (status_text, *empty_audio_updates())

        final_status = reporter.emit(i18n("Separation complete."), 1.0)
        if single_stem.strip():
            yield final_status, stems[0], None
        else:
            yield final_status, stems[0], stems[1]

    except Exception as e:
        reporter.emit(f"Roformer separation failed: {e}")
        raise RuntimeError(f"Roformer separation failed: {e}") from e

@track_presence("Performing MDXC Separationn")
def mdxc_separator(audio, model, out_format, segment_size, override_seg_size, overlap, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress(track_tqdm=True)):
    model_path = os.path.join(models_dir, model)
    reporter = StatusReporter(progress)

    def empty_audio_updates():
        return (gr.update(value=None), gr.update(value=None))

    try:
        status_text = reporter.emit(i18n("Preparing separation..."), 0.0)
        yield (status_text, *empty_audio_updates())

        if not os.path.exists(model_path):
            download_message = i18n("Downloading model: {model}...").format(model=model)
            status_text = reporter.emit(download_message)
            yield (status_text, *empty_audio_updates())
            gr.Info(i18n("This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.").format(model=model))

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=models_dir,
            output_dir=out_dir,
            output_format=out_format,
            use_autocast=use_autocast,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            output_single_stem=single_stem,
            mdxc_params={
                "segment_size": segment_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
            }
        )

        status_text = reporter.emit(i18n("Loading model..."), 0.2)
        yield (status_text, *empty_audio_updates())
        separator.load_model(model_filename=model)

        status_text = reporter.emit(i18n("Separating audio..."), 0.7)
        yield (status_text, *empty_audio_updates())
        separation = separator.separate(audio)

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        status_text = reporter.emit(i18n("Finalizing outputs..."), 0.9)
        yield (status_text, *empty_audio_updates())

        final_status = reporter.emit(i18n("Separation complete."), 1.0)
        if single_stem.strip():
            yield final_status, stems[0], None
        else:
            yield final_status, stems[0], stems[1]

    except Exception as e:
        reporter.emit(f"MDX23C separation failed: {e}")
        raise RuntimeError(f"MDX23C separation failed: {e}") from e

@track_presence("Performing MDX-NET Separation")
def mdxnet_separator(audio, model, out_format, hop_length, segment_size, denoise, overlap, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress(track_tqdm=True)):
    model_path = os.path.join(models_dir, model)
    reporter = StatusReporter(progress)

    def empty_audio_updates():
        return (gr.update(value=None), gr.update(value=None))

    try:
        status_text = reporter.emit(i18n("Preparing separation..."), 0.0)
        yield (status_text, *empty_audio_updates())

        if not os.path.exists(model_path):
            download_message = i18n("Downloading model: {model}...").format(model=model)
            status_text = reporter.emit(download_message)
            yield (status_text, *empty_audio_updates())
            gr.Info(i18n("This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.").format(model=model))

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=models_dir,
            output_dir=out_dir,
            output_format=out_format,
            use_autocast=use_autocast,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            output_single_stem=single_stem,
            mdx_params={
                "hop_length": hop_length,
                "segment_size": segment_size,
                "overlap": overlap,
                "batch_size": batch_size,
                "enable_denoise": denoise,
            }
        )

        status_text = reporter.emit(i18n("Loading model..."), 0.2)
        yield (status_text, *empty_audio_updates())
        separator.load_model(model_filename=model)

        status_text = reporter.emit(i18n("Separating audio..."), 0.7)
        yield (status_text, *empty_audio_updates())
        separation = separator.separate(audio)

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        status_text = reporter.emit(i18n("Finalizing outputs..."), 0.9)
        yield (status_text, *empty_audio_updates())

        final_status = reporter.emit(i18n("Separation complete."), 1.0)
        if single_stem.strip():
            yield final_status, stems[0], None
        else:
            yield final_status, stems[0], stems[1]

    except Exception as e:
        reporter.emit(f"MDX-NET separation failed: {e}")
        raise RuntimeError(f"MDX-NET separation failed: {e}") from e

@track_presence("Performing VR Arch Separation")
def vrarch_separator(audio, model, out_format, window_size, aggression, tta, post_process, post_process_threshold, high_end_process, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress(track_tqdm=True)):
    model_path = os.path.join(models_dir, model)
    reporter = StatusReporter(progress)

    def empty_audio_updates():
        return (gr.update(value=None), gr.update(value=None))

    try:
        status_text = reporter.emit(i18n("Preparing separation..."), 0.0)
        yield (status_text, *empty_audio_updates())

        if not os.path.exists(model_path):
            download_message = i18n("Downloading model: {model}...").format(model=model)
            status_text = reporter.emit(download_message)
            yield (status_text, *empty_audio_updates())
            gr.Info(i18n("This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.").format(model=model))

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=models_dir,
            output_dir=out_dir,
            output_format=out_format,
            use_autocast=use_autocast,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            output_single_stem=single_stem,
            vr_params={
                "batch_size": batch_size,
                "window_size": window_size,
                "aggression": aggression,
                "enable_tta": tta,
                "enable_post_process": post_process,
                "post_process_threshold": post_process_threshold,
                "high_end_process": high_end_process,
            }
        )

        status_text = reporter.emit(i18n("Loading model..."), 0.2)
        yield (status_text, *empty_audio_updates())
        separator.load_model(model_filename=model)

        status_text = reporter.emit(i18n("Separating audio..."), 0.7)
        yield (status_text, *empty_audio_updates())
        separation = separator.separate(audio)

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        status_text = reporter.emit(i18n("Finalizing outputs..."), 0.9)
        yield (status_text, *empty_audio_updates())

        final_status = reporter.emit(i18n("Separation complete."), 1.0)
        if single_stem.strip():
            yield final_status, stems[0], None
        else:
            yield final_status, stems[0], stems[1]

    except Exception as e:
        reporter.emit(f"VR ARCH separation failed: {e}")
        raise RuntimeError(f"VR ARCH separation failed: {e}") from e

@track_presence("Performing Demucs Separation")
def demucs_separator(audio, model, out_format, shifts, segment_size, segments_enabled, overlap, batch_size, norm_thresh, amp_thresh, progress=gr.Progress(track_tqdm=True)):
    model_path = os.path.join(models_dir, model)
    reporter = StatusReporter(progress)

    def empty_audio_updates():
        return tuple(gr.update(value=None) for _ in range(6))

    try:
        status_text = reporter.emit(i18n("Preparing separation..."), 0.0)
        yield (status_text, *empty_audio_updates())

        if not os.path.exists(model_path):
            download_message = i18n("Downloading model: {model}...").format(model=model)
            status_text = reporter.emit(download_message)
            yield (status_text, *empty_audio_updates())
            gr.Info(i18n("This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.").format(model=model))

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=models_dir,
            output_dir=out_dir,
            output_format=out_format,
            use_autocast=use_autocast,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            demucs_params={
                "batch_size": batch_size,
                "segment_size": segment_size,
                "shifts": shifts,
                "overlap": overlap,
                "segments_enabled": segments_enabled,
            }
        )

        status_text = reporter.emit(i18n("Loading model..."), 0.2)
        yield (status_text, *empty_audio_updates())
        separator.load_model(model_filename=model)

        status_text = reporter.emit(i18n("Separating audio..."), 0.7)
        yield (status_text, *empty_audio_updates())
        separation = separator.separate(audio)

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        status_text = reporter.emit(i18n("Finalizing outputs..."), 0.9)
        yield (status_text, *empty_audio_updates())

        final_status = reporter.emit(i18n("Separation complete."), 1.0)
        if model == "htdemucs_6s.yaml":
            yield (final_status, *stems[:6])
        else:
            yield final_status, stems[0], stems[1], stems[2], stems[3], None, None

    except Exception as e:
        reporter.emit(f"Demucs separation failed: {e}")
        raise RuntimeError(f"Demucs separation failed: {e}") from e

def update_stems(model):
    if model == "htdemucs_6s.yaml":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

@track_presence("Performing BS/Mel Roformer Batch Separation")
def roformer_batch(path_input, path_output, model_key, out_format, segment_size, override_seg_size, overlap, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress()):
    found_files.clear()
    logs.clear()
    roformer_model = roformer_models[model_key]
    model_path = os.path.join(models_dir, roformer_model)

    if not os.path.exists(model_path):
        gr.Info(f"This is the first time the {model_key} model is being used. The separation will take a little longer because the model needs to be downloaded.")

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i+1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    output_single_stem=single_stem,
                    mdxc_params={
                        "segment_size": segment_size,
                        "override_model_segment_size": override_seg_size,
                        "batch_size": batch_size,
                        "overlap": overlap,
                    }
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=roformer_model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"BS/Mel Roformer batch separation failed: {e}") from e
        
        progress(1.0, desc="Processing complete")
        return "\n".join(logs)

@track_presence("Performing MDXC Batch Separation")
def mdx23c_batch(path_input, path_output, model, out_format, segment_size, override_seg_size, overlap, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress()):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.")

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i+1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    output_single_stem=single_stem,
                    mdxc_params={
                        "segment_size": segment_size,
                        "override_model_segment_size": override_seg_size,
                        "batch_size": batch_size,
                        "overlap": overlap,
                    }
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"MDXC batch separation failed: {e}") from e
        
        progress(1.0, desc="Processing complete")
        return "\n".join(logs)

@track_presence("Performing MDX-NET Batch Separation")
def mdxnet_batch(path_input, path_output, model, out_format, hop_length, segment_size, denoise, overlap, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress()):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.")

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i+1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    output_single_stem=single_stem,
                    mdx_params={
                        "hop_length": hop_length,
                        "segment_size": segment_size,
                        "overlap": overlap,
                        "batch_size": batch_size,
                        "enable_denoise": denoise,
                    }
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"MDX-NET batch separation failed: {e}") from e
            
        progress(1.0, desc="Processing complete")
        return "\n".join(logs)

@track_presence("Performing VR Arch Batch Separation")
def vrarch_batch(path_input, path_output, model, out_format, window_size, aggression, tta, post_process, post_process_threshold, high_end_process, batch_size, norm_thresh, amp_thresh, single_stem, progress=gr.Progress()):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.")

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i+1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    output_single_stem=single_stem,
                    vr_params={
                        "batch_size": batch_size,
                        "window_size": window_size,
                        "aggression": aggression,
                        "enable_tta": tta,
                        "enable_post_process": post_process,
                        "post_process_threshold": post_process_threshold,
                        "high_end_process": high_end_process,
                    }
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"VR Arch batch separation failed: {e}") from e
            
        progress(1.0, desc="Processing complete")
        return "\n".join(logs)

@track_presence("Performing Demucs Batch Separation")
def demucs_batch(path_input, path_output, model, out_format, shifts, segment_size, segments_enabled, overlap, batch_size, norm_thresh, amp_thresh, progress=gr.Progress()):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded.")

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i+1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    demucs_params={
                        "batch_size": batch_size,
                        "segment_size": segment_size,
                        "shifts": shifts,
                        "overlap": overlap,
                        "segments_enabled": segments_enabled,
                    }
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"Demucs batch separation failed: {e}") from e
            
        progress(1.0, desc="Processing complete")
        return "\n".join(logs)


@track_presence("Performing Ensemble Separation")
def ensemble_separator(
    audio,
    selected_models,
    out_format,
    single_stem,
    normalization_threshold,
    amplification_threshold,
    roformer_segment_size,
    roformer_override_segment_size,
    roformer_overlap,
    roformer_batch_size,
    mdx23c_segment_size,
    mdx23c_override_segment_size,
    mdx23c_overlap,
    mdx23c_batch_size,
    mdxnet_hop_length,
    mdxnet_segment_size,
    mdxnet_denoise,
    mdxnet_overlap,
    mdxnet_batch_size,
    vrarch_window_size,
    vrarch_aggression,
    vrarch_tta,
    vrarch_post_process,
    vrarch_post_process_threshold,
    vrarch_high_end_process,
    vrarch_batch_size,
    demucs_shifts,
    demucs_segment_size,
    demucs_segments_enabled,
    demucs_overlap,
    demucs_batch_size,
    progress=gr.Progress(track_tqdm=True),
):
    temp_dirs = []
    reporter = StatusReporter(progress)

    def empty_audio_updates():
        return [gr.update(value=None, visible=False) for _ in range(6)]

    try:
        if not audio:
            raise gr.Error(i18n("Please provide an input audio file."))

        if not selected_models or len(selected_models) < 2:
            raise gr.Error(i18n("Select at least two models to perform an ensemble."))

        audio_path = audio
        if not os.path.exists(audio_path):
            raise gr.Error(i18n("The selected audio file could not be found."))

        output_format = (out_format or "wav").lower()
        single_stem_value = single_stem.strip()

        roformer_params = {
            "segment_size": int(roformer_segment_size),
            "override_segment_size": bool(roformer_override_segment_size),
            "overlap": int(roformer_overlap),
            "batch_size": int(roformer_batch_size),
        }
        mdx23c_params = {
            "segment_size": int(mdx23c_segment_size),
            "override_segment_size": bool(mdx23c_override_segment_size),
            "overlap": int(mdx23c_overlap),
            "batch_size": int(mdx23c_batch_size),
        }
        mdxnet_params = {
            "hop_length": int(mdxnet_hop_length),
            "segment_size": int(mdxnet_segment_size),
            "denoise": bool(mdxnet_denoise),
            "overlap": float(mdxnet_overlap),
            "batch_size": int(mdxnet_batch_size),
        }
        vrarch_params = {
            "window_size": int(vrarch_window_size),
            "aggression": int(vrarch_aggression),
            "tta": bool(vrarch_tta),
            "post_process": bool(vrarch_post_process),
            "post_process_threshold": float(vrarch_post_process_threshold),
            "high_end_process": bool(vrarch_high_end_process),
            "batch_size": int(vrarch_batch_size),
        }
        demucs_params = {
            "shifts": int(demucs_shifts),
            "segment_size": int(demucs_segment_size),
            "segments_enabled": bool(demucs_segments_enabled),
            "overlap": float(demucs_overlap),
            "batch_size": int(demucs_batch_size),
        }

        total_models = len(selected_models)
        total_steps = total_models + 2
        status_text = reporter.emit(i18n("Preparing ensemble..."), 0.0)
        yield (status_text, *empty_audio_updates())

        stem_maps = []
        model_displays = []

        for index, model_option in enumerate(selected_models):
            model_info = ensemble_model_map.get(model_option)
            if not model_info:
                raise gr.Error(i18n("Unknown model selected: {model}").format(model=model_option))

            model_displays.append(model_info["display"])
            status_text = reporter.emit(
                i18n("Separating with {model}...").format(model=model_option),
                index / total_steps,
            )
            yield (status_text, *empty_audio_updates())

            result = run_model_for_ensemble(
                audio_path,
                model_info,
                single_stem_value,
                normalization_threshold,
                amplification_threshold,
                roformer_params,
                mdx23c_params,
                mdxnet_params,
                vrarch_params,
                demucs_params,
            )

            temp_dirs.append(result["temp_dir"])
            stem_maps.append(result["stems"])

        status_text = reporter.emit(i18n("Combining stems..."), total_models / total_steps)
        yield (status_text, *empty_audio_updates())
        combined_outputs = combine_ensemble_stems(stem_maps)

        status_text = reporter.emit(
            i18n("Writing ensemble results..."), (total_models + 1) / total_steps
        )
        yield (status_text, *empty_audio_updates())
        ensemble_files = write_ensemble_outputs(
            combined_outputs,
            output_format,
            normalization_threshold,
            amplification_threshold,
        )

        final_status = reporter.emit(i18n("Ensemble complete"), 1.0)

        if ensemble_files:
            gr.Info(i18n("Ensemble completed using: {models}").format(models=", ".join(model_displays)))

        ensemble_label = i18n("Ensemble")
        updates = []
        for stem_name, file_path in ensemble_files:
            updates.append(gr.update(value=file_path, label=f"{stem_name} ({ensemble_label})", visible=True))

        for _ in range(len(ensemble_files), 6):
            updates.append(gr.update(value=None, visible=False))

        yield (final_status, *updates)

    except ValueError as e:
        reporter.emit(f"Ensemble error: {e}")
        raise gr.Error(i18n("Ensemble error: {message}").format(message=str(e)))
    except gr.Error as e:
        reporter.emit(f"Ensemble error: {e}")
        raise
    except Exception as e:
        reporter.emit(f"Ensemble separation failed: {e}")
        raise RuntimeError(f"Ensemble separation failed: {e}") from e
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

with gr.Blocks(theme = loadThemes.load_json() or "NoCrypt/miku", title = "🎵 UVR5 UI 🎵") as app:
    gr.Markdown("<h1> 🎵 UVR5 UI 🎵 </h1>")
    gr.Markdown(i18n("If you like UVR5 UI you can star my repo on [GitHub](https://github.com/Eddycrack864/UVR5-UI)"))
    gr.Markdown(i18n("Try UVR5 UI on Hugging Face with A100 [here](https://huggingface.co/spaces/TheStinger/UVR5_UI)"))
    all_configurable_inputs = []
    with gr.Tabs():
        with gr.TabItem("BS/Mel Roformer"):
            with gr.Row():
                roformer_model = gr.Dropdown(
                    label = i18n("Select the model"),
                    choices = list(roformer_models.keys()),
                    value = initial_settings.get("Roformer", {}).get("model", None),
                    interactive = True
                )
                roformer_output_format = gr.Dropdown(
                    label = i18n("Select the output format"),
                    choices = output_format,
                    value = initial_settings.get("Roformer", {}).get("output_format", None),
                    interactive = True
                )
            with gr.Accordion(i18n("Advanced settings"), open = False):
                with gr.Group():
                    with gr.Row():
                        roformer_segment_size = gr.Slider(
                            label = i18n("Segment size"),
                            info = i18n("Larger consumes more resources, but may give better results"),
                            minimum = 32,
                            maximum = 4000,
                            step = 32,
                            value = initial_settings.get("Roformer", {}).get("segment_size", 256),
                            interactive = True
                        )
                        roformer_override_segment_size = gr.Checkbox(
                            label = i18n("Override segment size"),
                            info = i18n("Override model default segment size instead of using the model default value"),
                            value = initial_settings.get("Roformer", {}).get("override_segment_size", False),
                            interactive = True
                        )
                    with gr.Row():
                        roformer_overlap = gr.Slider(
                            label = i18n("Overlap"),
                            info = i18n("Amount of overlap between prediction windows"),
                            minimum = 2,
                            maximum = 10,
                            step = 1,
                            value = initial_settings.get("Roformer", {}).get("overlap", 8),
                            interactive = True
                        )
                        roformer_batch_size = gr.Slider(
                            label = i18n("Batch size"),
                            info = i18n("Larger consumes more RAM but may process slightly faster"),
                            minimum = 1,
                            maximum = 16,
                            step = 1,
                            value = initial_settings.get("Roformer", {}).get("batch_size", 1),
                            interactive = True
                        )
                    with gr.Row():
                        roformer_normalization_threshold = gr.Slider(
                            label = i18n("Normalization threshold"),
                            info = i18n("The threshold for audio normalization"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("Roformer", {}).get("normalization_threshold", 0.9),
                            interactive = True
                        )
                        roformer_amplification_threshold = gr.Slider(
                            label = i18n("Amplification threshold"),
                            info = i18n("The threshold for audio amplification"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("Roformer", {}).get("amplification_threshold", 0.7),
                            interactive = True
                        )
                    with gr.Row():
                        roformer_single_stem = gr.Textbox(
                            label = i18n("Output only single stem"),
                            placeholder = i18n("Write the stem you want, check the stems of each model on Leaderboard. e.g. Instrumental"),
                            value = initial_settings.get("Roformer", {}).get("single_stem", ""),
                            interactive = True
                        )
            with gr.Row():
                roformer_audio = gr.Audio(
                    label = i18n("Input audio"),
                    type = "filepath",
                    interactive = True
                )
            with gr.Accordion(i18n("Separation by link"), open = False):
                with gr.Row():
                    roformer_link = gr.Textbox(
                        label = i18n("Link"),
                        placeholder = i18n("Paste the link here"),
                        value = initial_settings.get("Roformer", {}).get("link", ""),
                        interactive = True
                    )
                with gr.Row():
                    gr.Markdown(i18n("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)"))
                with gr.Row():
                    roformer_download_button = gr.Button(
                        i18n("Download!"),
                        variant = "primary"
                    )

            roformer_download_button.click(download_audio, [roformer_link], [roformer_audio])

            with gr.Accordion(i18n("Batch separation"), open = False):
                with gr.Row():
                    roformer_input_path = gr.Textbox(
                        label = i18n("Input path"),
                        placeholder = i18n("Place the input path here"),
                        value = initial_settings.get("Roformer", {}).get("input_path", ""),
                        interactive = True
                    )
                    roformer_output_path = gr.Textbox(
                        label = i18n("Output path"),
                        placeholder = i18n("Place the output path here"),
                        value = initial_settings.get("Roformer", {}).get("output_path", ""),
                        interactive = True
                    )
                with gr.Row():
                    roformer_bath_button = gr.Button(i18n("Separate!"), variant = "primary")
                with gr.Row():
                    roformer_info = gr.Textbox(
                        label = i18n("Output information"),
                        interactive = False
                    )

            components["Roformer"] = {
                        "model": roformer_model,
                        "output_format": roformer_output_format,
                        "segment_size": roformer_segment_size,
                        "override_segment_size": roformer_override_segment_size,
                        "overlap": roformer_overlap,
                        "batch_size": roformer_batch_size,
                        "normalization_threshold": roformer_normalization_threshold,
                        "amplification_threshold": roformer_amplification_threshold,
                        "single_stem": roformer_single_stem,
                        "link": roformer_link,
                        "input_path": roformer_input_path,
                        "output_path": roformer_output_path
                    }
            all_configurable_inputs.extend(components["Roformer"].values())
            
            roformer_bath_button.click(roformer_batch, [roformer_input_path, roformer_output_path, roformer_model, roformer_output_format, roformer_segment_size, roformer_override_segment_size, roformer_overlap, roformer_batch_size, roformer_normalization_threshold, roformer_amplification_threshold, roformer_single_stem], [roformer_info])

            with gr.Row():
                roformer_button = gr.Button(i18n("Separate!"), variant = "primary")
            with gr.Row():
                roformer_status = gr.Textbox(
                    label = i18n("Status log"),
                    interactive = False,
                    lines = 6,
                    value = ""
                )
            with gr.Row():
                roformer_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = i18n("Stem 1"),
                    type = "filepath"
                )
                roformer_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = i18n("Stem 2"),
                    type = "filepath"
                )

            roformer_button.click(
                roformer_separator,
                [
                    roformer_audio,
                    roformer_model,
                    roformer_output_format,
                    roformer_segment_size,
                    roformer_override_segment_size,
                    roformer_overlap,
                    roformer_batch_size,
                    roformer_normalization_threshold,
                    roformer_amplification_threshold,
                    roformer_single_stem,
                ],
                [roformer_status, roformer_stem1, roformer_stem2],
            )

        with gr.TabItem("MDX23C"):
            with gr.Row():
                mdx23c_model = gr.Dropdown(
                    label = i18n("Select the model"),
                    choices = mdx23c_models,
                    value = initial_settings.get("MDX23C", {}).get("model", None),
                    interactive = True
                )
                mdx23c_output_format = gr.Dropdown(
                    label = i18n("Select the output format"),
                    choices = output_format,
                    value = initial_settings.get("MDX23C", {}).get("output_format", None),
                    interactive = True
                )
            with gr.Accordion(i18n("Advanced settings"), open = False):
                with gr.Group():
                    with gr.Row():
                        mdx23c_segment_size = gr.Slider(
                            minimum = 32,
                            maximum = 4000,
                            step = 32,
                            label = i18n("Segment size"),
                            info = i18n("Larger consumes more resources, but may give better results"),
                            value = initial_settings.get("MDX23C", {}).get("segment_size", 256),
                            interactive = True
                        )
                        mdx23c_override_segment_size = gr.Checkbox(
                            label = i18n("Override segment size"),
                            info = i18n("Override model default segment size instead of using the model default value"),
                            value = initial_settings.get("MDX23C", {}).get("override_segment_size", False),
                            interactive = True
                        )
                    with gr.Row():
                        mdx23c_overlap = gr.Slider(
                            minimum = 2,
                            maximum = 50,
                            step = 1,
                            label = i18n("Overlap"),
                            info = i18n("Amount of overlap between prediction windows"),
                            value = initial_settings.get("MDX23C", {}).get("overlap", 8),
                            interactive = True
                        )
                        mdx23c_batch_size = gr.Slider(
                            label = i18n("Batch size"),
                            info = i18n("Larger consumes more RAM but may process slightly faster"),
                            minimum = 1,
                            maximum = 16,
                            step = 1,
                            value = initial_settings.get("MDX23C", {}).get("batch_size", 1),
                            interactive = True
                        )
                    with gr.Row():
                        mdx23c_normalization_threshold = gr.Slider(
                            label = i18n("Normalization threshold"),
                            info = i18n("The threshold for audio normalization"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("MDX23C", {}).get("normalization_threshold", 0.9),
                            interactive = True
                        )
                        mdx23c_amplification_threshold = gr.Slider(
                            label = i18n("Amplification threshold"),
                            info = i18n("The threshold for audio amplification"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("MDX23C", {}).get("amplification_threshold", 0.7),
                            interactive = True
                        )
                    with gr.Row():
                        mdx23c_single_stem = gr.Textbox(
                            label = i18n("Output only single stem"),
                            placeholder = i18n("Write the stem you want, check the stems of each model on Leaderboard. e.g. Instrumental"),
                            value = initial_settings.get("MDX23C", {}).get("single_stem", ""),
                            interactive = True
                        )
            with gr.Row():
                mdx23c_audio = gr.Audio(
                    label = i18n("Input audio"),
                    type = "filepath",
                    interactive = True
                )
            with gr.Accordion(i18n("Separation by link"), open = False):
                with gr.Row():
                    mdx23c_link = gr.Textbox(
                        label = i18n("Link"),
                        placeholder = i18n("Paste the link here"),
                        value = initial_settings.get("MDX23C", {}).get("link", ""),
                        interactive = True
                    )
                with gr.Row():
                    gr.Markdown(i18n("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)"))
                with gr.Row():
                    mdx23c_download_button = gr.Button(
                        i18n("Download!"),
                        variant = "primary"
                    )

            mdx23c_download_button.click(download_audio, [mdx23c_link], [mdx23c_audio])

            with gr.Accordion(i18n("Batch separation"), open = False):
                with gr.Row():
                    mdx23c_input_path = gr.Textbox(
                        label = i18n("Input path"),
                        placeholder = i18n("Place the input path here"),
                        value = initial_settings.get("MDX23C", {}).get("input_path", ""),
                        interactive = True
                    )
                    mdx23c_output_path = gr.Textbox(
                        label = i18n("Output path"),
                        placeholder = i18n("Place the output path here"),
                        value = initial_settings.get("MDX23C", {}).get("output_path", ""),
                        interactive = True
                    )
                with gr.Row():
                    mdx23c_bath_button = gr.Button(i18n("Separate!"), variant = "primary")
                with gr.Row():
                    mdx23c_info = gr.Textbox(
                        label = i18n("Output information"),
                        interactive = False
                    )

            components["MDX23C"] = {
                        "model": mdx23c_model,
                        "output_format": mdx23c_output_format,
                        "segment_size": mdx23c_segment_size,
                        "override_segment_size": mdx23c_override_segment_size,
                        "overlap": mdx23c_overlap,
                        "batch_size": mdx23c_batch_size,
                        "normalization_threshold": mdx23c_normalization_threshold,
                        "amplification_threshold": mdx23c_amplification_threshold,
                        "single_stem": mdx23c_single_stem,
                        "link": mdx23c_link,
                        "input_path": mdx23c_input_path,
                        "output_path": mdx23c_output_path
                    }
            all_configurable_inputs.extend(components["MDX23C"].values())
            
            mdx23c_bath_button.click(mdx23c_batch, [mdx23c_input_path, mdx23c_output_path, mdx23c_model, mdx23c_output_format, mdx23c_segment_size, mdx23c_override_segment_size, mdx23c_overlap, mdx23c_batch_size, mdx23c_normalization_threshold, mdx23c_amplification_threshold, mdx23c_single_stem], [mdx23c_info])

            with gr.Row():
                mdx23c_button = gr.Button(i18n("Separate!"), variant = "primary")
            with gr.Row():
                mdx23c_status = gr.Textbox(
                    label = i18n("Status log"),
                    interactive = False,
                    lines = 6,
                    value = ""
                )
            with gr.Row():
                mdx23c_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = i18n("Stem 1"),
                    type = "filepath"
                )
                mdx23c_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = i18n("Stem 2"),
                    type = "filepath"
                )

            mdx23c_button.click(
                mdxc_separator,
                [
                    mdx23c_audio,
                    mdx23c_model,
                    mdx23c_output_format,
                    mdx23c_segment_size,
                    mdx23c_override_segment_size,
                    mdx23c_overlap,
                    mdx23c_batch_size,
                    mdx23c_normalization_threshold,
                    mdx23c_amplification_threshold,
                    mdx23c_single_stem,
                ],
                [mdx23c_status, mdx23c_stem1, mdx23c_stem2],
            )
                
        with gr.TabItem("MDX-NET"):
            with gr.Row():
                mdxnet_model = gr.Dropdown(
                    label = i18n("Select the model"),
                    choices = mdxnet_models,
                    value = initial_settings.get("MDX-NET", {}).get("model", None),
                    interactive = True
                )
                mdxnet_output_format = gr.Dropdown(
                    label = i18n("Select the output format"),
                    choices = output_format,
                    value = initial_settings.get("MDX-NET", {}).get("output_format", None),
                    interactive = True
                )
            with gr.Accordion(i18n("Advanced settings"), open = False):
                with gr.Group():
                    with gr.Row():
                        mdxnet_hop_length = gr.Slider(
                            label = i18n("Hop length"),
                            info = i18n("Usually called stride in neural networks; only change if you know what you're doing"),
                            minimum = 32,
                            maximum = 2048,
                            step = 32,
                            value = initial_settings.get("MDX-NET", {}).get("hop_length", 1024),
                            interactive = True
                        )
                        mdxnet_segment_size = gr.Slider(
                            minimum = 32,
                            maximum = 4000,
                            step = 32,
                            label = i18n("Segment size"),
                            info = i18n("Larger consumes more resources, but may give better results"),
                            value = initial_settings.get("MDX-NET", {}).get("segment_size", 256),
                            interactive = True
                        )
                        mdxnet_denoise = gr.Checkbox(
                            label = i18n("Denoise"),
                            info = i18n("Enable denoising during separation"),
                            value = initial_settings.get("MDX-NET", {}).get("denoise", True),
                            interactive = True
                        )
                    with gr.Row():
                        mdxnet_overlap = gr.Slider(
                            label = i18n("Overlap"),
                            info = i18n("Amount of overlap between prediction windows"),
                            minimum = 0.001,
                            maximum = 0.999,
                            step = 0.001,
                            value = initial_settings.get("MDX-NET", {}).get("overlap", 0.25),
                            interactive = True
                        )
                        mdxnet_batch_size = gr.Slider(
                            label = i18n("Batch size"),
                            info = i18n("Larger consumes more RAM but may process slightly faster"),
                            minimum = 1,
                            maximum = 16,
                            step = 1,
                            value = initial_settings.get("MDX-NET", {}).get("batch_size", 1),
                            interactive = True
                        )
                    with gr.Row():
                        mdxnet_normalization_threshold = gr.Slider(
                            label = i18n("Normalization threshold"),
                            info = i18n("The threshold for audio normalization"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("MDX-NET", {}).get("normalization_threshold", 0.9),
                            interactive = True
                        )
                        mdxnet_amplification_threshold = gr.Slider(
                            label = i18n("Amplification threshold"),
                            info = i18n("The threshold for audio amplification"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("MDX-NET", {}).get("amplification_threshold", 0.7),
                            interactive = True
                        )
                    with gr.Row():
                        mdxnet_single_stem = gr.Textbox(
                            label = i18n("Output only single stem"),
                            placeholder = i18n("Write the stem you want, check the stems of each model on Leaderboard. e.g. Instrumental"),
                            value = initial_settings.get("MDX-NET", {}).get("single_stem", ""),
                            interactive = True
                        )
            with gr.Row():
                mdxnet_audio = gr.Audio(
                    label = i18n("Input audio"),
                    type = "filepath",
                    interactive = True
                )
            with gr.Accordion(i18n("Separation by link"), open = False):
                with gr.Row():
                    mdxnet_link = gr.Textbox(
                        label = i18n("Link"),
                        placeholder = i18n("Paste the link here"),
                        value = initial_settings.get("MDX-NET", {}).get("link", ""),
                        interactive = True
                    )
                with gr.Row():
                    gr.Markdown(i18n("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)"))
                with gr.Row():
                    mdxnet_download_button = gr.Button(
                        i18n("Download!"),
                        variant = "primary"
                    )
                
            mdxnet_download_button.click(download_audio, [mdxnet_link], [mdxnet_audio])

            with gr.Accordion(i18n("Batch separation"), open = False):
                with gr.Row():
                    mdxnet_input_path = gr.Textbox(
                        label = i18n("Input path"),
                        placeholder = i18n("Place the input path here"),
                        value = initial_settings.get("MDX-NET", {}).get("input_path", ""),
                        interactive = True
                    )
                    mdxnet_output_path = gr.Textbox(
                        label = i18n("Output path"),
                        placeholder = i18n("Place the output path here"),
                        value = initial_settings.get("MDX-NET", {}).get("output_path", ""),
                        interactive = True
                    )
                with gr.Row():
                    mdxnet_bath_button = gr.Button(i18n("Separate!"), variant = "primary")
                with gr.Row():
                    mdxnet_info = gr.Textbox(
                        label = i18n("Output information"),
                        interactive = False
                    )

            components["MDX-NET"] = {
                        "model": mdxnet_model,
                        "output_format": mdxnet_output_format,
                        "hop_length": mdxnet_hop_length,
                        "segment_size": mdxnet_segment_size,
                        "denoise": mdxnet_denoise,
                        "overlap": mdxnet_overlap,
                        "batch_size": mdxnet_batch_size,
                        "normalization_threshold": mdxnet_normalization_threshold,
                        "amplification_threshold": mdxnet_amplification_threshold,
                        "single_stem": mdxnet_single_stem,
                        "link": mdxnet_link,
                        "input_path": mdxnet_input_path,
                        "output_path": mdxnet_output_path
                    }
            all_configurable_inputs.extend(components["MDX-NET"].values())

            mdxnet_bath_button.click(mdxnet_batch, [mdxnet_input_path, mdxnet_output_path, mdxnet_model, mdxnet_output_format, mdxnet_hop_length, mdxnet_segment_size, mdxnet_denoise, mdxnet_overlap, mdxnet_batch_size, mdxnet_normalization_threshold, mdxnet_amplification_threshold, mdxnet_single_stem], [mdxnet_info])

            with gr.Row():
                mdxnet_button = gr.Button(i18n("Separate!"), variant = "primary")
            with gr.Row():
                mdxnet_status = gr.Textbox(
                    label = i18n("Status log"),
                    interactive = False,
                    lines = 6,
                    value = ""
                )
            with gr.Row():
                mdxnet_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = i18n("Stem 1"),
                    type = "filepath"
                )
                mdxnet_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = i18n("Stem 2"),
                    type = "filepath"
                )

            mdxnet_button.click(
                mdxnet_separator,
                [
                    mdxnet_audio,
                    mdxnet_model,
                    mdxnet_output_format,
                    mdxnet_hop_length,
                    mdxnet_segment_size,
                    mdxnet_denoise,
                    mdxnet_overlap,
                    mdxnet_batch_size,
                    mdxnet_normalization_threshold,
                    mdxnet_amplification_threshold,
                    mdxnet_single_stem,
                ],
                [mdxnet_status, mdxnet_stem1, mdxnet_stem2],
            )

        with gr.TabItem("VR ARCH"):
            with gr.Row():
                vrarch_model = gr.Dropdown(
                    label = i18n("Select the model"),
                    choices = vrarch_models,
                    value = initial_settings.get("VR Arch", {}).get("model", None),
                    interactive = True
                )
                vrarch_output_format = gr.Dropdown(
                    label = i18n("Select the output format"),
                    choices = output_format,
                    value = initial_settings.get("VR Arch", {}).get("output_format", None),
                    interactive = True
                )
            with gr.Accordion(i18n("Advanced settings"), open = False):
                with gr.Group():
                    with gr.Row():
                        vrarch_window_size = gr.Slider(
                            label = i18n("Window size"),
                            info = i18n("Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality"),
                            minimum=320,
                            maximum=1024,
                            step=32,
                            value = initial_settings.get("VR Arch", {}).get("window_size", 512),
                            interactive = True
                        )
                        vrarch_agression = gr.Slider(
                            minimum = 1,
                            maximum = 50,
                            step = 1,
                            label = i18n("Agression"),
                            info = i18n("Intensity of primary stem extraction"),
                            value = initial_settings.get("VR Arch", {}).get("aggression", 5),
                            interactive = True
                        )
                        vrarch_tta = gr.Checkbox(
                            label = i18n("TTA"),
                            info = i18n("Enable Test-Time-Augmentation; slow but improves quality"),
                            value = initial_settings.get("VR Arch", {}).get("tta", True),
                            visible = True,
                            interactive = True
                        )
                    with gr.Row():
                        vrarch_post_process = gr.Checkbox(
                            label = i18n("Post process"),
                            info = i18n("Identify leftover artifacts within vocal output; may improve separation for some songs"),
                            value = initial_settings.get("VR Arch", {}).get("post_process", False),
                            visible = True,
                            interactive = True
                        )
                        vrarch_post_process_threshold = gr.Slider(
                            label = i18n("Post process threshold"),
                            info = i18n("Threshold for post-processing"),
                            minimum = 0.1,
                            maximum = 0.3,
                            step = 0.1,
                            value = initial_settings.get("VR Arch", {}).get("post_process_threshold", 0.2),
                            interactive = True
                        )
                    with gr.Row():
                        vrarch_high_end_process = gr.Checkbox(
                            label = i18n("High end process"),
                            info = i18n("Mirror the missing frequency range of the output"),
                            value = initial_settings.get("VR Arch", {}).get("high_end_process", False),
                            visible = True,
                            interactive = True,
                        )
                        vrarch_batch_size = gr.Slider(
                            label = i18n("Batch size"),
                            info = i18n("Larger consumes more RAM but may process slightly faster"),
                            minimum = 1,
                            maximum = 16,
                            step = 1,
                            value = initial_settings.get("VR Arch", {}).get("batch_size", 1),
                            interactive = True
                        )
                    with gr.Row():
                        vrarch_normalization_threshold = gr.Slider(
                            label = i18n("Normalization threshold"),
                            info = i18n("The threshold for audio normalization"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("VR Arch", {}).get("normalization_threshold", 0.9),
                            interactive = True
                        )
                        vrarch_amplification_threshold = gr.Slider(
                            label = i18n("Amplification threshold"),
                            info = i18n("The threshold for audio amplification"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("VR Arch", {}).get("amplification_threshold", 0.7),
                            interactive = True
                        )
                    with gr.Row():
                        vrarch_single_stem = gr.Textbox(
                            label = i18n("Output only single stem"),
                            placeholder = i18n("Write the stem you want, check the stems of each model on Leaderboard. e.g. Instrumental"),
                            value = initial_settings.get("VR Arch", {}).get("single_stem", ""),
                            interactive = True
                        )
            with gr.Row():
                vrarch_audio = gr.Audio(
                    label = i18n("Input audio"),
                    type = "filepath",
                    interactive = True
                )
            with gr.Accordion(i18n("Separation by link"), open = False):
                with gr.Row():
                    vrarch_link = gr.Textbox(
                        label = i18n("Link"),
                        placeholder = i18n("Paste the link here"),
                        value = initial_settings.get("VR Arch", {}).get("link", ""),
                        interactive = True
                    )
                with gr.Row():
                    gr.Markdown(i18n("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)"))
                with gr.Row():
                    vrarch_download_button = gr.Button(
                        i18n("Download!"),
                        variant = "primary"
                )

            vrarch_download_button.click(download_audio, [vrarch_link], [vrarch_audio])
            
            with gr.Accordion(i18n("Batch separation"), open = False):
                with gr.Row():
                    vrarch_input_path = gr.Textbox(
                        label = i18n("Input path"),
                        placeholder = i18n("Place the input path here"),
                        value = initial_settings.get("VR Arch", {}).get("input_path", ""),
                        interactive = True
                    )
                    vrarch_output_path = gr.Textbox(
                        label = i18n("Output path"),
                        placeholder = i18n("Place the output path here"),
                        value = initial_settings.get("VR Arch", {}).get("output_path", ""),
                        interactive = True
                    )
                with gr.Row():
                    vrarch_bath_button = gr.Button(i18n("Separate!"), variant = "primary")
                with gr.Row():
                    vrarch_info = gr.Textbox(
                        label = i18n("Output information"),
                        interactive = False
                    )

            components["VR Arch"] = {
                        "model": vrarch_model,
                        "output_format": vrarch_output_format,
                        "window_size": vrarch_window_size,
                        "aggression": vrarch_agression,
                        "tta": vrarch_tta,
                        "post_process": vrarch_post_process,
                        "post_process_threshold": vrarch_post_process_threshold,
                        "high_end_process": vrarch_high_end_process,
                        "batch_size": vrarch_batch_size,
                        "normalization_threshold": vrarch_normalization_threshold,
                        "amplification_threshold": vrarch_amplification_threshold,
                        "single_stem": vrarch_single_stem,
                        "link": vrarch_link,
                        "input_path": vrarch_input_path,
                        "output_path": vrarch_output_path
                    }
            all_configurable_inputs.extend(components["VR Arch"].values())

            vrarch_bath_button.click(vrarch_batch, [vrarch_input_path, vrarch_output_path, vrarch_model, vrarch_output_format, vrarch_window_size, vrarch_agression, vrarch_tta, vrarch_post_process, vrarch_post_process_threshold, vrarch_high_end_process, vrarch_batch_size, vrarch_normalization_threshold, vrarch_amplification_threshold, vrarch_single_stem], [vrarch_info])

            with gr.Row():
                vrarch_button = gr.Button(i18n("Separate!"), variant = "primary")
            with gr.Row():
                vrarch_status = gr.Textbox(
                    label = i18n("Status log"),
                    interactive = False,
                    lines = 6,
                    value = ""
                )
            with gr.Row():
                vrarch_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 1")
                )
                vrarch_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 2")
                )

            vrarch_button.click(
                vrarch_separator,
                [
                    vrarch_audio,
                    vrarch_model,
                    vrarch_output_format,
                    vrarch_window_size,
                    vrarch_agression,
                    vrarch_tta,
                    vrarch_post_process,
                    vrarch_post_process_threshold,
                    vrarch_high_end_process,
                    vrarch_batch_size,
                    vrarch_normalization_threshold,
                    vrarch_amplification_threshold,
                    vrarch_single_stem,
                ],
                [vrarch_status, vrarch_stem1, vrarch_stem2],
            )

        with gr.TabItem("Demucs"):
            with gr.Row():
                demucs_model = gr.Dropdown(
                    label = i18n("Select the model"),
                    choices = demucs_models,
                    value = initial_settings.get("Demucs", {}).get("model", None),
                    interactive = True
                )
                demucs_output_format = gr.Dropdown(
                    label = i18n("Select the output format"),
                    choices = output_format,
                    value = initial_settings.get("Demucs", {}).get("output_format", None),
                    interactive = True
                )
            with gr.Accordion(i18n("Advanced settings"), open = False):
                with gr.Group():
                    with gr.Row():
                        demucs_shifts = gr.Slider(
                            label = i18n("Shifts"),
                            info = i18n("Number of predictions with random shifts, higher = slower but better quality"),
                            minimum = 1,
                            maximum = 20,
                            step = 1,
                            value = initial_settings.get("Demucs", {}).get("shifts", 2),
                            interactive = True
                        )
                        demucs_segment_size = gr.Slider(
                            label = i18n("Segment size"),
                            info = i18n("Size of segments into which the audio is split. Higher = slower but better quality"),
                            minimum = 1,
                            maximum = 100,
                            step = 1,
                            value = initial_settings.get("Demucs", {}).get("segment_size", 40),
                            interactive = True
                        )
                        demucs_segments_enabled = gr.Checkbox(
                            label = i18n("Segment-wise processing"),
                            info = i18n("Enable segment-wise processing"),
                            value = initial_settings.get("Demucs", {}).get("segments_enabled", True),
                            interactive = True
                        )
                    with gr.Row():
                        demucs_overlap = gr.Slider(
                            label = i18n("Overlap"),
                            info = i18n("Overlap between prediction windows. Higher = slower but better quality"),
                            minimum=0.001,
                            maximum=0.999,
                            step=0.001,
                            value = initial_settings.get("Demucs", {}).get("overlap", 0.25),
                            interactive = True
                        )
                        demucs_batch_size = gr.Slider(
                            label = i18n("Batch size"),
                            info = i18n("Larger consumes more RAM but may process slightly faster"),
                            minimum = 1,
                            maximum = 16,
                            step = 1,
                            value = initial_settings.get("Demucs", {}).get("batch_size", 1),
                            interactive = True
                        )
                    with gr.Row():
                        demucs_normalization_threshold = gr.Slider(
                            label = i18n("Normalization threshold"),
                            info = i18n("The threshold for audio normalization"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("Demucs", {}).get("normalization_threshold", 0.9),
                            interactive = True
                        )
                        demucs_amplification_threshold = gr.Slider(
                            label = i18n("Amplification threshold"),
                            info = i18n("The threshold for audio amplification"),
                            minimum = 0.1,
                            maximum = 1,
                            step = 0.1,
                            value = initial_settings.get("Demucs", {}).get("amplification_threshold", 0.7),
                            interactive = True
                        )
            with gr.Row():
                demucs_audio = gr.Audio(
                    label = i18n("Input audio"),
                    type = "filepath",
                    interactive = True
                )
            with gr.Accordion(i18n("Separation by link"), open = False):
                with gr.Row():
                    demucs_link = gr.Textbox(
                        label = i18n("Link"),
                        placeholder = i18n("Paste the link here"),
                        value = initial_settings.get("Demucs", {}).get("link", ""),
                        interactive = True
                    )
                with gr.Row():
                    gr.Markdown(i18n("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)"))
                with gr.Row():
                    demucs_download_button = gr.Button(
                        i18n("Download!"),
                        variant = "primary"
                    )

            demucs_download_button.click(download_audio, [demucs_link], [demucs_audio])

            with gr.Accordion(i18n("Batch separation"), open = False):
                with gr.Row():
                    demucs_input_path = gr.Textbox(
                        label = i18n("Input path"),
                        placeholder = i18n("Place the input path here"),
                        value = initial_settings.get("Demucs", {}).get("input_path", ""),
                        interactive = True
                    )
                    demucs_output_path = gr.Textbox(
                        label = i18n("Output path"),
                        placeholder = i18n("Place the output path here"),
                        value = initial_settings.get("Demucs", {}).get("output_path", ""),
                        interactive = True
                    )
                with gr.Row():
                    demucs_bath_button = gr.Button(i18n("Separate!"), variant = "primary")
                with gr.Row():
                    demucs_info = gr.Textbox(
                        label = i18n("Output information"),
                        interactive = False
                    )

            components["Demucs"] = {
                        "model": demucs_model,
                        "output_format": demucs_output_format,
                        "shifts": demucs_shifts,
                        "segment_size": demucs_segment_size,
                        "segments_enabled": demucs_segments_enabled,
                        "overlap": demucs_overlap,
                        "batch_size": demucs_batch_size,
                        "normalization_threshold": demucs_normalization_threshold,
                        "amplification_threshold": demucs_amplification_threshold,
                        "link": demucs_link,
                        "input_path": demucs_input_path,
                        "output_path": demucs_output_path
                    }
            all_configurable_inputs.extend(components["Demucs"].values())

            demucs_bath_button.click(demucs_batch, [demucs_input_path, demucs_output_path, demucs_model, demucs_output_format, demucs_shifts, demucs_segment_size, demucs_segments_enabled, demucs_overlap, demucs_batch_size, demucs_normalization_threshold, demucs_amplification_threshold], [demucs_info])

            with gr.Row():
                demucs_button = gr.Button(i18n("Separate!"), variant = "primary")
            with gr.Row():
                demucs_status = gr.Textbox(
                    label = i18n("Status log"),
                    interactive = False,
                    lines = 6,
                    value = ""
                )
            with gr.Row():
                demucs_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 1")
                )
                demucs_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 2")
                )
            with gr.Row():
                demucs_stem3 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 3")
                )
                demucs_stem4 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 4")
                )
            with gr.Row(visible=False) as stem6:
                demucs_stem5 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 5")
                )
                demucs_stem6 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 6")
                )

            demucs_model.change(update_stems, inputs=[demucs_model], outputs=stem6)

            demucs_button.click(
                demucs_separator,
                [
                    demucs_audio,
                    demucs_model,
                    demucs_output_format,
                    demucs_shifts,
                    demucs_segment_size,
                    demucs_segments_enabled,
                    demucs_overlap,
                    demucs_batch_size,
                    demucs_normalization_threshold,
                    demucs_amplification_threshold,
                ],
                [demucs_status, demucs_stem1, demucs_stem2, demucs_stem3, demucs_stem4, demucs_stem5, demucs_stem6],
            )

        with gr.TabItem(i18n("Ensemble")):
            gr.Markdown(i18n("Combine multiple models into an averaged result. Select at least two models and adjust per-architecture options below."))
            with gr.Row():
                ensemble_models = gr.CheckboxGroup(
                    label = i18n("Models to ensemble"),
                    info = i18n("Choose two or more models to average their outputs"),
                    choices = ensemble_model_choices,
                    value = initial_settings.get("Ensemble", {}).get("models", []),
                    interactive = True
                )
                ensemble_output_format = gr.Dropdown(
                    label = i18n("Select the output format"),
                    choices = output_format,
                    value = initial_settings.get("Ensemble", {}).get("output_format", None),
                    interactive = True
                )
            with gr.Row():
                ensemble_single_stem = gr.Textbox(
                    label = i18n("Output only single stem"),
                    placeholder = i18n("Write the stem you want, check the stems of each model on Leaderboard. e.g. Instrumental"),
                    value = initial_settings.get("Ensemble", {}).get("single_stem", ""),
                    interactive = True
                )
            with gr.Row():
                ensemble_normalization_threshold = gr.Slider(
                    label = i18n("Normalization threshold"),
                    info = i18n("The threshold for audio normalization"),
                    minimum = 0.1,
                    maximum = 1,
                    step = 0.1,
                    value = initial_settings.get("Ensemble", {}).get("normalization_threshold", 0.9),
                    interactive = True
                )
                ensemble_amplification_threshold = gr.Slider(
                    label = i18n("Amplification threshold"),
                    info = i18n("The threshold for audio amplification"),
                    minimum = 0.1,
                    maximum = 1,
                    step = 0.1,
                    value = initial_settings.get("Ensemble", {}).get("amplification_threshold", 0.7),
                    interactive = True
                )
            with gr.Accordion(i18n("Advanced settings"), open = False):
                gr.Markdown(i18n("Settings apply only to the selected models of each architecture."))
                with gr.Tabs():
                    with gr.TabItem(i18n("BS/Mel Roformer")):
                        with gr.Row():
                            ensemble_roformer_segment_size = gr.Slider(
                                label = i18n("Segment size"),
                                minimum = 32,
                                maximum = 4000,
                                step = 32,
                                value = initial_settings.get("Ensemble", {}).get("roformer_segment_size", 256),
                                interactive = True,
                            )
                            ensemble_roformer_override_segment_size = gr.Checkbox(
                                label = i18n("Override segment size"),
                                value = initial_settings.get("Ensemble", {}).get("roformer_override_segment_size", False),
                                interactive = True,
                            )
                        with gr.Row():
                            ensemble_roformer_overlap = gr.Slider(
                                label = i18n("Overlap"),
                                minimum = 2,
                                maximum = 10,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("roformer_overlap", 8),
                                interactive = True,
                            )
                            ensemble_roformer_batch_size = gr.Slider(
                                label = i18n("Batch size"),
                                minimum = 1,
                                maximum = 16,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("roformer_batch_size", 1),
                                interactive = True,
                            )

                    with gr.TabItem("MDX23C"):
                        with gr.Row():
                            ensemble_mdx23c_segment_size = gr.Slider(
                                label = i18n("Segment size"),
                                minimum = 32,
                                maximum = 4000,
                                step = 32,
                                value = initial_settings.get("Ensemble", {}).get("mdx23c_segment_size", 256),
                                interactive = True,
                            )
                            ensemble_mdx23c_override_segment_size = gr.Checkbox(
                                label = i18n("Override segment size"),
                                value = initial_settings.get("Ensemble", {}).get("mdx23c_override_segment_size", False),
                                interactive = True,
                            )
                        with gr.Row():
                            ensemble_mdx23c_overlap = gr.Slider(
                                label = i18n("Overlap"),
                                minimum = 2,
                                maximum = 50,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("mdx23c_overlap", 8),
                                interactive = True,
                            )
                            ensemble_mdx23c_batch_size = gr.Slider(
                                label = i18n("Batch size"),
                                minimum = 1,
                                maximum = 16,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("mdx23c_batch_size", 1),
                                interactive = True,
                            )

                    with gr.TabItem("MDX-NET"):
                        with gr.Row():
                            ensemble_mdxnet_hop_length = gr.Slider(
                                label = i18n("Hop length"),
                                minimum = 32,
                                maximum = 2048,
                                step = 32,
                                value = initial_settings.get("Ensemble", {}).get("mdxnet_hop_length", 1024),
                                interactive = True,
                            )
                            ensemble_mdxnet_segment_size = gr.Slider(
                                label = i18n("Segment size"),
                                minimum = 32,
                                maximum = 4000,
                                step = 32,
                                value = initial_settings.get("Ensemble", {}).get("mdxnet_segment_size", 256),
                                interactive = True,
                            )
                        with gr.Row():
                            ensemble_mdxnet_denoise = gr.Checkbox(
                                label = i18n("Denoise"),
                                value = initial_settings.get("Ensemble", {}).get("mdxnet_denoise", True),
                                interactive = True,
                            )
                            ensemble_mdxnet_overlap = gr.Slider(
                                label = i18n("Overlap"),
                                minimum = 0.0,
                                maximum = 0.99,
                                step = 0.01,
                                value = initial_settings.get("Ensemble", {}).get("mdxnet_overlap", 0.25),
                                interactive = True,
                            )
                            ensemble_mdxnet_batch_size = gr.Slider(
                                label = i18n("Batch size"),
                                minimum = 1,
                                maximum = 16,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("mdxnet_batch_size", 1),
                                interactive = True,
                            )

                    with gr.TabItem(i18n("VR Arch")):
                        with gr.Row():
                            ensemble_vrarch_window_size = gr.Slider(
                                label = i18n("Window size"),
                                minimum = 320,
                                maximum = 1024,
                                step = 32,
                                value = initial_settings.get("Ensemble", {}).get("vrarch_window_size", 512),
                                interactive = True,
                            )
                            ensemble_vrarch_aggression = gr.Slider(
                                label = i18n("Agression"),
                                minimum = 1,
                                maximum = 50,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("vrarch_aggression", 5),
                                interactive = True,
                            )
                            ensemble_vrarch_tta = gr.Checkbox(
                                label = i18n("TTA"),
                                value = initial_settings.get("Ensemble", {}).get("vrarch_tta", True),
                                interactive = True,
                            )
                        with gr.Row():
                            ensemble_vrarch_post_process = gr.Checkbox(
                                label = i18n("Post process"),
                                value = initial_settings.get("Ensemble", {}).get("vrarch_post_process", False),
                                interactive = True,
                            )
                            ensemble_vrarch_post_process_threshold = gr.Slider(
                                label = i18n("Post process threshold"),
                                minimum = 0.1,
                                maximum = 0.3,
                                step = 0.1,
                                value = initial_settings.get("Ensemble", {}).get("vrarch_post_process_threshold", 0.2),
                                interactive = True,
                            )
                            ensemble_vrarch_high_end_process = gr.Checkbox(
                                label = i18n("High end process"),
                                value = initial_settings.get("Ensemble", {}).get("vrarch_high_end_process", False),
                                interactive = True,
                            )
                        with gr.Row():
                            ensemble_vrarch_batch_size = gr.Slider(
                                label = i18n("Batch size"),
                                minimum = 1,
                                maximum = 16,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("vrarch_batch_size", 1),
                                interactive = True,
                            )

                    with gr.TabItem("Demucs"):
                        with gr.Row():
                            ensemble_demucs_shifts = gr.Slider(
                                label = i18n("Shifts"),
                                minimum = 1,
                                maximum = 20,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("demucs_shifts", 2),
                                interactive = True,
                            )
                            ensemble_demucs_segment_size = gr.Slider(
                                label = i18n("Segment size"),
                                minimum = 1,
                                maximum = 100,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("demucs_segment_size", 40),
                                interactive = True,
                            )
                            ensemble_demucs_segments_enabled = gr.Checkbox(
                                label = i18n("Segment-wise processing"),
                                value = initial_settings.get("Ensemble", {}).get("demucs_segments_enabled", True),
                                interactive = True,
                            )
                        with gr.Row():
                            ensemble_demucs_overlap = gr.Slider(
                                label = i18n("Overlap"),
                                minimum = 0.001,
                                maximum = 0.999,
                                step = 0.001,
                                value = initial_settings.get("Ensemble", {}).get("demucs_overlap", 0.25),
                                interactive = True,
                            )
                            ensemble_demucs_batch_size = gr.Slider(
                                label = i18n("Batch size"),
                                minimum = 1,
                                maximum = 16,
                                step = 1,
                                value = initial_settings.get("Ensemble", {}).get("demucs_batch_size", 1),
                                interactive = True,
                            )
            with gr.Row():
                ensemble_audio = gr.Audio(
                    label = i18n("Input audio"),
                    type = "filepath",
                    interactive = True
                )
            with gr.Row():
                ensemble_button = gr.Button(i18n("Run ensemble"), variant = "primary")
            with gr.Row():
                ensemble_status = gr.Textbox(
                    label = i18n("Status log"),
                    interactive = False,
                    lines = 6,
                    value = ""
                )
            with gr.Row():
                ensemble_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 1"),
                    visible = False
                )
                ensemble_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 2"),
                    visible = False
                )
            with gr.Row():
                ensemble_stem3 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 3"),
                    visible = False
                )
                ensemble_stem4 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 4"),
                    visible = False
                )
            with gr.Row():
                ensemble_stem5 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 5"),
                    visible = False
                )
                ensemble_stem6 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = i18n("Stem 6"),
                    visible = False
                )

            ensemble_outputs = [ensemble_status, ensemble_stem1, ensemble_stem2, ensemble_stem3, ensemble_stem4, ensemble_stem5, ensemble_stem6]

            components["Ensemble"] = {
                        "models": ensemble_models,
                        "output_format": ensemble_output_format,
                        "single_stem": ensemble_single_stem,
                        "normalization_threshold": ensemble_normalization_threshold,
                        "amplification_threshold": ensemble_amplification_threshold,
                        "roformer_segment_size": ensemble_roformer_segment_size,
                        "roformer_override_segment_size": ensemble_roformer_override_segment_size,
                        "roformer_overlap": ensemble_roformer_overlap,
                        "roformer_batch_size": ensemble_roformer_batch_size,
                        "mdx23c_segment_size": ensemble_mdx23c_segment_size,
                        "mdx23c_override_segment_size": ensemble_mdx23c_override_segment_size,
                        "mdx23c_overlap": ensemble_mdx23c_overlap,
                        "mdx23c_batch_size": ensemble_mdx23c_batch_size,
                        "mdxnet_hop_length": ensemble_mdxnet_hop_length,
                        "mdxnet_segment_size": ensemble_mdxnet_segment_size,
                        "mdxnet_denoise": ensemble_mdxnet_denoise,
                        "mdxnet_overlap": ensemble_mdxnet_overlap,
                        "mdxnet_batch_size": ensemble_mdxnet_batch_size,
                        "vrarch_window_size": ensemble_vrarch_window_size,
                        "vrarch_aggression": ensemble_vrarch_aggression,
                        "vrarch_tta": ensemble_vrarch_tta,
                        "vrarch_post_process": ensemble_vrarch_post_process,
                        "vrarch_post_process_threshold": ensemble_vrarch_post_process_threshold,
                        "vrarch_high_end_process": ensemble_vrarch_high_end_process,
                        "vrarch_batch_size": ensemble_vrarch_batch_size,
                        "demucs_shifts": ensemble_demucs_shifts,
                        "demucs_segment_size": ensemble_demucs_segment_size,
                        "demucs_segments_enabled": ensemble_demucs_segments_enabled,
                        "demucs_overlap": ensemble_demucs_overlap,
                        "demucs_batch_size": ensemble_demucs_batch_size
                    }
            all_configurable_inputs.extend(components["Ensemble"].values())

            ensemble_button.click(
                ensemble_separator,
                [
                    ensemble_audio,
                    ensemble_models,
                    ensemble_output_format,
                    ensemble_single_stem,
                    ensemble_normalization_threshold,
                    ensemble_amplification_threshold,
                    ensemble_roformer_segment_size,
                    ensemble_roformer_override_segment_size,
                    ensemble_roformer_overlap,
                    ensemble_roformer_batch_size,
                    ensemble_mdx23c_segment_size,
                    ensemble_mdx23c_override_segment_size,
                    ensemble_mdx23c_overlap,
                    ensemble_mdx23c_batch_size,
                    ensemble_mdxnet_hop_length,
                    ensemble_mdxnet_segment_size,
                    ensemble_mdxnet_denoise,
                    ensemble_mdxnet_overlap,
                    ensemble_mdxnet_batch_size,
                    ensemble_vrarch_window_size,
                    ensemble_vrarch_aggression,
                    ensemble_vrarch_tta,
                    ensemble_vrarch_post_process,
                    ensemble_vrarch_post_process_threshold,
                    ensemble_vrarch_high_end_process,
                    ensemble_vrarch_batch_size,
                    ensemble_demucs_shifts,
                    ensemble_demucs_segment_size,
                    ensemble_demucs_segments_enabled,
                    ensemble_demucs_overlap,
                    ensemble_demucs_batch_size,
                ],
                ensemble_outputs,
            )

        with gr.TabItem(i18n("Leaderboard")):
            with gr.Group():
                with gr.Row(equal_height=True):
                    list_filter = gr.Dropdown(
                        label = i18n("List filter"),
                        info = i18n("Filter and sort the model list by stem"),
                        choices = ["vocals", "instrumental", "reverb", "echo", "noise", "crowd", "dry", "aspiration", "male", "woodwinds", "kick", "drums", "bass", "guitar", "piano", "other"],
                        value = lambda : None
                    )
                    list_button = gr.Button(i18n("Show list!"), variant = "primary")
            output_list = gr.HTML(label = i18n("Leaderboard"))

            list_button.click(leaderboard, inputs=list_filter, outputs=output_list)

        with gr.TabItem(i18n("Themes")):
            themes_select = gr.Dropdown(
                label = i18n("Theme"),
                info = i18n("Select the theme you want to use. (Requires restarting the App)"),
                choices = loadThemes.get_list(),
                value = loadThemes.read_json(),
                interactive = True
            )

            themes_select.change(
                fn = loadThemes.select_theme,
                inputs = themes_select,
                outputs = []
            )

        with gr.TabItem(i18n("Settings")):
            with gr.Accordion(i18n("Language selector"), open = False):
                selected_language = gr.Dropdown(
                    label = i18n("Language"),
                    info = i18n("Select the language you want to use. (Requires restarting the App)"),
                    value = get_language_settings(),
                    choices = ["Language automatically detected by system"] + i18n._get_available_languages(),
                    interactive = True
                )

                selected_language.change(
                    fn = save_lang_settings,
                    inputs = [selected_language],
                    outputs=[]
                )
            with gr.Accordion(i18n("Alternative model downloader"), open = False):
                download_method = gr.Dropdown(
                    label = i18n("Download method"),
                    info = i18n("Select the download method you want to use. (Must have it installed)"),
                    value = lambda : None,
                    choices = ["wget", "curl"],
                    interactive = True
                )
                model_key_json = gr.Dropdown(
                    label = i18n("Model to download"),
                    info = i18n("Select the model to download using the selected method"),
                    value = lambda : None,
                    choices = list(roformer_models.keys()) + mdx23c_models + mdxnet_models + vrarch_models,
                    interactive = True
                )
                alternative_model_downloader_button = gr.Button(i18n("Download!"), variant = "primary")
                alternative_model_downloader_output = gr.Textbox(
                    label = i18n("Output information"),
                    interactive = False
                )

                alternative_model_downloader_button.click(alternative_model_downloader, [download_method, model_key_json], [alternative_model_downloader_output])

            with gr.Accordion(i18n("Separation settings management"), open = False):
                gr.Markdown(i18n("Save your current separation parameter settings or reset them to the application defaults"))
                with gr.Row():
                    save_settings_button = gr.Button(i18n("Save current settings"), variant = "primary")
                    reset_settings_button = gr.Button(i18n("Reset settings to default"), variant = "primary")

                save_settings_button.click(save_current_settings, all_configurable_inputs, None)
                reset_settings_button.click(reset_settings_to_default, None, all_configurable_inputs)

        with gr.TabItem(i18n("Credits")):
            gr.Markdown(
                """
                UVR5 UI created by **[Eddycrack 864](https://github.com/Eddycrack864).** Join **[AI HUB](https://discord.gg/aihub)** community.
                * python-audio-separator by [beveradb](https://github.com/beveradb).
                * Special thanks to [Ilaria](https://github.com/TheStingerX) for hosting this space and help.
                * Thanks to [Mikus](https://github.com/cappuch) for the help with the code.
                * Thanks to [Nick088](https://huggingface.co/Nick088) for the help to fix roformers.
                * Thanks to [yt_dlp](https://github.com/yt-dlp/yt-dlp) devs.
                * Separation by link source code and improvements by [NeoDev](https://github.com/TheNeodev).
                * Thanks to [ArisDev](https://github.com/aris-py) for porting UVR5 UI to Kaggle and improvements.
                * Thanks to [Bebra777228](https://github.com/Bebra777228)'s code for guiding me to improve my code.
                * Thanks to Nick088, MrM0dZ, Ryouko-Yamanda65777, lucinamari, perariroswe, Enes, Léo, the_undead0 and Storm for helping translate UVR5 UI.
                * Thanks to vadigr123 for creating the images for the Discord Rich Presence.
                
                You can donate to the original UVR5 project here:
                [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/uvr5)
                """
            )

app.launch(
    share=args.share,
    favicon_path="assets/favicon.ico",
    server_name="",
    server_port=args.listen_port,
    inbrowser=args.open
)