import os
import sys
import re
from easy_functions import (format_time,
                            get_input_length,
                            get_video_details,
                            show_video,
                            count_audio_file,
                            g_colab)
import contextlib
import shutil
import subprocess
import time
from IPython.display import Audio, Image, clear_output, display
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import configparser
import utils
from img2video_batch import check_and_convert_image_to_video
import datetime

input_media = "in/1.png"
input_vocal = utils.find_audio('in')
ffmpeg_path = 'ffmpeg.exe'

try:
    if input_media:
        print("找到图片文件，开始转换成视频")
        input_media = check_and_convert_image_to_video('in',input_media)
    else:
        print("在 'in' 文件夹中没有发现图片或视频文件。")

    if not input_vocal:
        print("在 'in' 文件夹中没有发现音频文件。")
except ValueError as err:
    print(err)


# retrieve variables from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

video_file = f"in\\1.mp4"
vocal_file = f"in\\1.wav"

quality = config['OPTIONS']['quality']
output_height = config['OPTIONS']['output_height']
wav2lip_version = config['OPTIONS']['wav2lip_version']
use_previous_tracking_data = config['OPTIONS']['use_previous_tracking_data']
nosmooth = config.getboolean('OPTIONS', 'nosmooth')
U = config.getint('PADDING', 'U')
D = config.getint('PADDING', 'D')
L = config.getint('PADDING', 'L')
R = config.getint('PADDING', 'R')
size = config.getfloat('MASK', 'size')
feathering = config.getint('MASK', 'feathering')
mouth_tracking = config.getboolean('MASK', 'mouth_tracking')
debug_mask = config.getboolean('MASK', 'debug_mask')
batch_process = config.getboolean('OTHER', 'batch_process')
output_suffix = config['OTHER']['output_suffix']
include_settings_in_suffix = config.getboolean('OTHER', 'include_settings_in_suffix')
if g_colab():
	preview_input = config.getboolean('OTHER', 'preview_input')
else:
	preview_input = False
preview_settings = config.getboolean('OTHER', 'preview_settings')
frame_to_preview = config.getint('OTHER', 'frame_to_preview')

working_directory = os.getcwd()


start_time = time.time()

video_file = video_file.strip('\"')
vocal_file = vocal_file.strip('\"')

# check video_file exists
if video_file=='':
  sys.exit(f'video_file cannot be blank')

if not os.path.exists(video_file):
  sys.exit(f'Could not find file: {video_file}')

if wav2lip_version=="Wav2Lip_GAN":
  checkpoint_path = os.path.join(working_directory,'checkpoints','Wav2Lip_GAN.pth')
else:
  checkpoint_path = os.path.join(working_directory,'checkpoints','Wav2Lip.pth')

if feathering == 3:
  feathering = 5
if feathering == 2:
  feathering = 3

resolution_scale = 1
res_custom = False
if output_height == 'half resolution':
  resolution_scale = 2
elif output_height == 'full resolution':
  resolution_scale = 1
else:
  res_custom = True
  resolution_scale = 3

in_width, in_height, in_fps, in_length = get_video_details(video_file)
out_height = round(in_height / resolution_scale)

if res_custom:
  out_height = int(output_height)
fps_for_static_image = 30


if output_suffix == '' and not include_settings_in_suffix:
  sys.exit('Current suffix settings will overwrite your input video! Please add a suffix or tick include_settings_in_suffix')

frame_to_preview = max(frame_to_preview -1,0)

if include_settings_in_suffix:
  if wav2lip_version=="Wav2Lip_GAN":
    output_suffix = f'{output_suffix}_GAN'
  output_suffix = f'{output_suffix}_{quality}'
  if output_height != 'full resolution':
    output_suffix = f'{output_suffix}_{out_height}'
  if nosmooth:
    output_suffix = f'{output_suffix}_nosmooth1'
  else:
    output_suffix = f'{output_suffix}_nosmooth0'
  if U!=0 or D!=0 or L!=0 or R!=0:
    output_suffix = f'{output_suffix}_pads-'
    if U!=0:
      output_suffix = f'{output_suffix}U{U}'
    if D!=0:
      output_suffix = f'{output_suffix}D{D}'
    if L!=0:
      output_suffix = f'{output_suffix}L{L}'
    if R!=0:
      output_suffix = f'{output_suffix}R{R}'
  if quality != 'fast':
    output_suffix = f'{output_suffix}_mask-S{size}F{feathering}'
    if mouth_tracking:
      output_suffix = f'{output_suffix}_mt'
    if debug_mask:
      output_suffix = f'{output_suffix}_debug'



rescaleFactor = str(round(1 // resolution_scale))
pad_up = str(round(U * resolution_scale))
pad_down = str(round(D * resolution_scale))
pad_left = str(round(L * resolution_scale))
pad_right = str(round(R * resolution_scale))
################################################################################


######################### reconstruct input paths ##############################
# Extract each part of the path
folder, filename_with_extension = os.path.split(video_file)
filename, file_type = os.path.splitext(filename_with_extension)

# Extract filenumber if it exists
filenumber_match = re.search(r"\d+$", filename)
if filenumber_match: # if there is a filenumber - extract it
  filenumber = str(filenumber_match.group())
  filenamenonumber = re.sub(r"\d+$", "", filename)
else: # if there is no filenumber - make it blank
  filenumber = ""
  filenamenonumber = filename

# if vocal_file is blank - use the video as audio
if vocal_file == "":
  vocal_file = video_file
# if not, check that the vocal_file file exists
else:
  if not os.path.exists(vocal_file):
    sys.exit(f'Could not find file: {vocal_file}')

# Extract each part of the path
audio_folder, audio_filename_with_extension = os.path.split(vocal_file)
audio_filename, audio_file_type = os.path.splitext(audio_filename_with_extension)

# Extract filenumber if it exists
audio_filenumber_match = re.search(r"\d+$", audio_filename)
if audio_filenumber_match: # if there is a filenumber - extract it
  audio_filenumber = str(audio_filenumber_match.group())
  audio_filenamenonumber = re.sub(r"\d+$", "", audio_filename)
else: # if there is no filenumber - make it blank
  audio_filenumber = ""
  audio_filenamenonumber = audio_filename
################################################################################

# set process_failed to False so that it may be set to True if one or more processings fail
process_failed = False


temp_output = os.path.join(working_directory,'temp','output.mp4')
temp_folder = os.path.join(working_directory,'temp')

last_input_video = None
last_input_audio = None

#--------------------------Batch processing loop-------------------------------!
count_audios = count_audio_file("in")+1
for i in range(1,count_audios):
    filenumber = 1
    # construct input_video
    input_video = os.path.join(folder, str(filenumber) + file_type)
    input_videofile = os.path.basename(input_video)

    # construct input_audio
    input_audio = os.path.join(audio_folder, str(i) + audio_file_type)
    input_audiofile = os.path.basename(input_audio)

    # construct output_video
    output_video = os.path.join(folder, f'{i}-out.mp4')
    #如果文件已经存在，直接跳过下一个
    if os.path.exists(output_video):
      continue

    output_videofile = os.path.basename(output_video)

    # remove last outputs
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        os.makedirs('temp', exist_ok=True)


    last_input_video = input_video
    last_input_audio = input_audio
    shutil.copy(input_video, temp_folder)
    shutil.copy(input_audio, temp_folder)

    #rename temp file to include padding or else changing padding does nothing
    temp_input_video =  os.path.join(temp_folder,input_videofile)
    renamed_temp_input_video = os.path.join(temp_folder, str(U)+str(D)+str(L)+str(R) + input_videofile)
    shutil.copy(temp_input_video, renamed_temp_input_video)
    temp_input_video = renamed_temp_input_video
    temp_input_videofile = os.path.basename(renamed_temp_input_video)
    temp_input_audio =  os.path.join(temp_folder,input_audiofile)

    #trim video if it's longer than the audio
    video_length = get_input_length(temp_input_video)
    audio_length = get_input_length(temp_input_audio)


    if video_length > audio_length:

        trimmed_video_path = os.path.join(temp_folder,"trimmed_" + temp_input_videofile)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                ffmpeg_extract_subclip(temp_input_video, 0, audio_length, targetname=trimmed_video_path)
        temp_input_video = trimmed_video_path

    #check if face detection has already happened on this clip
    last_detected_face = os.path.join(working_directory,'last_detected_face.pkl')
    if os.path.isfile('last_file.txt'):
        with open('last_file.txt', 'r') as file:
            last_file = file.readline()
        if last_file != temp_input_video or use_previous_tracking_data == False:
            print("准备删除：",last_detected_face)
            if os.path.isfile(last_detected_face):
                os.remove(last_detected_face)

    #----------------------------Process the inputs!-----------------------------!
    print(f"处理中{' preview of' if preview_settings else ''} "
        f"{input_videofile} 使用 {input_audiofile} 作为音频文件")

    #execute Wav2Lip & upscaler

    cmd = [
      "python", "inference_batch.py",
      "--face", temp_input_video,
      "--audio", temp_input_audio,
      "--outfile", temp_output,
      "--pads", str(pad_up), str(pad_down), str(pad_left), str(pad_right),
      "--checkpoint_path", checkpoint_path,
      "--out_height", str(out_height),
      "--fullres", str(resolution_scale),
      "--quality", quality,
      "--mask_dilation", str(size),
      "--mask_feathering", str(feathering),
      "--nosmooth", str(nosmooth),
      "--debug_mask", str(debug_mask),
      "--preview_settings", str(preview_settings),
      "--mouth_tracking", str(mouth_tracking)
    ]

    # Run the command
    subprocess.run(cmd)


    #rename temp file and move to correct directory
    print("last...",temp_output)
    if os.path.isfile(temp_output):
        if os.path.isfile(output_video):
          os.remove(output_video)
        shutil.copy(temp_output, output_video)
        #show output video
        with open('last_file.txt', 'w') as f:
          f.write(temp_input_video)
        print(f"{output_video} 已经成功生成")

        #end processing timer and format the time it took
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_setup_time = format_time(elapsed_time)
        print(f"执行时间: {formatted_setup_time}")

    else:
        print(f"Processing failed! :( see line above 👆")
        if not g_colab:
            print('也许这个在您的系统上不兼容，您可能最好去aibl论坛看看:')
            print('https://www.aibl.vip')
        process_failed = True



