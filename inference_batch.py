print('\rloading torch       ', end='')
import torch
print('\rloading numpy       ', end='')
import numpy as np
print('\rloading Image       ', end='')
from PIL import Image
print('\rloading argparse    ', end='')
import argparse
print('\rloading math        ', end='')
import math
print('\rloading os          ', end='')
import os
print('\rloading subprocess  ', end='')
import subprocess
print('\rloading pickle      ', end='')
import pickle
print('\rloading cv2         ', end='')
import cv2
print('\rloading audio       ', end='')
import audio
print('\rloading Wav2Lip     ', end='')
from models import Wav2Lip
print('\rRloading RetinaFace ', end='')
from batch_face import RetinaFace
print('\rloading re          ', end='')
import re
print('\rloading partial     ', end='')
from functools import partial
print('\rloading tqdm        ', end='')
from tqdm import tqdm
print('\rloading warnings    ', end='')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')
print('\rloading upscale     ', end='')
from enhance import upscale
print('\rloading load_sr     ', end='')
from enhance import load_sr
print('\rloading load_model  ', end='')
from easy_functions import load_model

print('\rimports loaded!     ')

device = 'cuda'
parser = argparse.ArgumentParser(description='使用Wav2Lip模型在野外同步视频的推理代码')

parser.add_argument('--checkpoint_path', type=str,
                    help='要加载权重的保存检查点的名称', required=True)

parser.add_argument('--segmentation_path', type=str, default="checkpoints/face_segmentation.pth",
                    help='分割网络的保存检查点的名称', required=False)

parser.add_argument('--face', type=str,
                    help='包含要使用的人脸的视频/图像的文件路径', required=True)
parser.add_argument('--audio', type=str,
                    help='要用作原始音频源的视频/音频文件的文件路径', required=True)
parser.add_argument('--outfile', type=str, help='保存结果的视频路径。参见默认值示例。',
                    default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
                    help='如果为True，则仅对推理使用第一帧视频', default=False)
parser.add_argument('--fps', type=float, help='只能在输入为静态图像时指定（默认值：25）',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='填充（上，下，左，右）。请调整以至少包括下巴')

parser.add_argument('--wav2lip_batch_size', type=int, help='Wav2Lip模型的批处理大小', default=1)

parser.add_argument('--out_height', default=480, type=int,
                    help='输出视频的高度。在480或720时获得最佳结果')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1,  0, -1],
                    help='将视频裁剪到较小的区域（上，下，左，右）。在resize_factor和rotate参数之后应用。'
                         '如果有多个面，这将很有用。-1意味着该值将根据高度和宽度自动推断')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='为面部指定常数边界框。仅在面部检测失败时使用，可能仅在面部移动不大时有效。'
                         '语法：（上，下，左，右）')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='有时从手机拍摄的视频可能会翻转90度。如果为True，将视频向右翻转90度。'
                         '如果得到翻转的结果，请使用此选项')

parser.add_argument('--nosmooth', type=str, default=False,
                    help='防止在短时间窗口内对面部检测进行平滑处理')

parser.add_argument('--no_seg', default=False, action='store_true',
                    help='防止使用面部分割')

parser.add_argument('--no_sr', default=False, action='store_true',
                    help='防止使用超分辨率')

parser.add_argument('--sr_model', type=str, default='gfpgan',
                    help='升频器的名称 - gfpgan或RestoreFormer', required=False)

parser.add_argument('--fullres', default=3, type=int,
                    help='仅用于确定是否使用全分辨率，以便不需要进行调整大小')

parser.add_argument('--debug_mask', type=str, default=False,
                    help='使背景灰度以更好地查看掩码')

parser.add_argument('--preview_settings', type=str, default=False,
                    help='仅处理一帧')

parser.add_argument('--mouth_tracking', type=str, default=False,
                    help='在每一帧中跟踪口罩')

parser.add_argument('--mask_dilation', default=150, type=float,
                    help='口罩周围的大小', required=False)

parser.add_argument('--mask_feathering', default=151, type=int,
                    help='口罩周围的羽化程度', required=False)

parser.add_argument('--quality', type=str, help='选择Fast，Improved，Enhanced和Experimental之间', 
                                default='Fast')            

with open(os.path.join('checkpoints', 'predictor.pkl'), 'rb') as f:
    predictor = pickle.load(f)


with open(os.path.join('checkpoints','mouth_detector.pkl'), 'rb') as f:
    mouth_detector = pickle.load(f)

#creating variables to prevent failing when a face isn't detected
kernel = last_mask = x = y = w = h = None
def Experimental(img, original_img,run_params): 
  global kernel, last_mask, x, y, w, h  # Add last_mask to global variables
  
   # Convert color space from BGR to RGB if necessary 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  original_img  = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

  if str(args.debug_mask) == 'True':
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

   # Detect face 
  faces = mouth_detector(img) 
  if len(faces) == 0: 
     if last_mask is not None: 
       last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
       mask = last_mask  # use the last successful mask 
     else: 
       cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
       return img, None 
  else: 
      face = faces[0] 
      shape = predictor(img, face) 
  
      # Get points for mouth 
      mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

      # Calculate bounding box dimensions
      x, y, w, h = cv2.boundingRect(mouth_points)

      # Set kernel size as a fraction of bounding box size
      kernel_size = int(max(w, h) * args.mask_dilation)
      upscale_kernel_size = int(max(w, h) * max(args.mask_dilation, 2.5))

      # Create kernels
      kernel = np.ones((kernel_size, kernel_size), np.uint8)
      upscale_kernel = np.ones((upscale_kernel_size, upscale_kernel_size), np.uint8)

      # Create binary mask for mouth 
      mask = np.zeros(img.shape[:2], dtype=np.uint8) 
      cv2.fillConvexPoly(mask, mouth_points, 255)

      last_mask = mask  # Update last_mask with the new mask
  
  # Dilate the mask for upscaling
  upscale_dilated_mask = cv2.dilate(mask, upscale_kernel)
  dilated_mask = cv2.dilate(mask, kernel)

  # Find contours in the dilated mask
  contours, _ = cv2.findContours(upscale_dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find bounding box coordinates for each contour
  for contour in contours:
    x_dilated, y_dilated, w_dilated, h_dilated = cv2.boundingRect(contour)

    # Crop the image to the bounding box of the dilated mask
    cropped_img = img[y_dilated:y_dilated+h_dilated, x_dilated:x_dilated+w_dilated]

    # Save the cropped image here
    #cv2.imwrite('temp/cp.jpg', cropped_img)

    # Upscale the cropped image
    upscaled_img = upscale(cropped_img, run_params)

    #cv2.imwrite('temp/ucp.jpg', upscaled_img)

    if str(args.debug_mask) == 'True':
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Paste the upscaled image back onto the original image
    img[y_dilated:y_dilated+h_dilated, x_dilated:x_dilated+w_dilated] = upscaled_img


  # Calculate distance transform of dilated mask
  dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

  # Normalize distance transform
  cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

  # Convert normalized distance transform to binary mask and convert it to uint8
  _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
  masked_diff = masked_diff.astype(np.uint8)
  
  if not args.mask_feathering == 0:
    blur = args.mask_feathering
    # Set blur size as a fraction of bounding box size
    blur = int(max(w, h) * blur)  # 10% of bounding box size
    if blur % 2 == 0:  # Ensure blur size is odd
      blur += 1
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

  # Convert numpy arrays to PIL Images
  input1 = Image.fromarray(img)
  input2 = Image.fromarray(original_img)

  # Convert mask to single channel where pixel values are from the alpha channel of the current mask
  mask = Image.fromarray(masked_diff)

  # Ensure images are the same size
  assert input1.size == input2.size == mask.size

  # Paste input1 onto input2 using the mask
  input2.paste(input1, (0,0), mask)

  # Convert the final PIL Image back to a numpy array
  input2 = np.array(input2)

  cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
  
  return input2, mask

def create_tracked_mask(img, original_img): 
  global kernel, last_mask, x, y, w, h  # Add last_mask to global variables
  
   # Convert color space from BGR to RGB if necessary 
  cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
  cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img) 
  
   # Detect face 
  faces = mouth_detector(img) 
  if len(faces) == 0: 
     if last_mask is not None: 
       last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
       mask = last_mask  # use the last successful mask 
     else: 
       cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
       return img, None 
  else: 
      face = faces[0] 
      shape = predictor(img, face) 
  
      # Get points for mouth 
      mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

      # Calculate bounding box dimensions
      x, y, w, h = cv2.boundingRect(mouth_points)

      # Set kernel size as a fraction of bounding box size
      kernel_size = int(max(w, h) * args.mask_dilation)
      #if kernel_size % 2 == 0:  # Ensure kernel size is odd
        #kernel_size += 1

      # Create kernel
      kernel = np.ones((kernel_size, kernel_size), np.uint8)

      # Create binary mask for mouth 
      mask = np.zeros(img.shape[:2], dtype=np.uint8) 
      cv2.fillConvexPoly(mask, mouth_points, 255)

      last_mask = mask  # Update last_mask with the new mask
  
  # Dilate the mask
  dilated_mask = cv2.dilate(mask, kernel)

  # Calculate distance transform of dilated mask
  dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

  # Normalize distance transform
  cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

  # Convert normalized distance transform to binary mask and convert it to uint8
  _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
  masked_diff = masked_diff.astype(np.uint8)
  
  #make sure blur is an odd number
  blur = args.mask_feathering
  if blur % 2 == 0:
    blur += 1
  # Set blur size as a fraction of bounding box size
  blur = int(max(w, h) * blur)  # 10% of bounding box size
  if blur % 2 == 0:  # Ensure blur size is odd
    blur += 1
  masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

  # Convert numpy arrays to PIL Images
  input1 = Image.fromarray(img)
  input2 = Image.fromarray(original_img)

  # Convert mask to single channel where pixel values are from the alpha channel of the current mask
  mask = Image.fromarray(masked_diff)

  # Ensure images are the same size
  assert input1.size == input2.size == mask.size

  # Paste input1 onto input2 using the mask
  input2.paste(input1, (0,0), mask)

  # Convert the final PIL Image back to a numpy array
  input2 = np.array(input2)

  #input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
  cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
  
  return input2, mask

def create_mask(img, original_img): 
  global kernel, last_mask, x, y, w, h  # Add last_mask to global variables
  
   # Convert color space from BGR to RGB if necessary 
  cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
  cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

  if last_mask is not None: 
      last_mask = np.array(last_mask)  # Convert PIL Image to numpy array
      last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
      mask = last_mask  # use the last successful mask 
      mask = Image.fromarray(mask)

  else:
    # Detect face 
    faces = mouth_detector(img) 
    if len(faces) == 0: 
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
        return img, None 
    else: 
        face = faces[0] 
        shape = predictor(img, face) 
    
        # Get points for mouth 
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

        # Calculate bounding box dimensions
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Set kernel size as a fraction of bounding box size
        kernel_size = int(max(w, h) * args.mask_dilation)
        #if kernel_size % 2 == 0:  # Ensure kernel size is odd
          #kernel_size += 1

        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create binary mask for mouth 
        mask = np.zeros(img.shape[:2], dtype=np.uint8) 
        cv2.fillConvexPoly(mask, mouth_points, 255)

        # Dilate the mask
        dilated_mask = cv2.dilate(mask, kernel)

        # Calculate distance transform of dilated mask
        dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

        # Normalize distance transform
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        # Convert normalized distance transform to binary mask and convert it to uint8
        _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
        masked_diff = masked_diff.astype(np.uint8)

        if not args.mask_feathering == 0:
          blur = args.mask_feathering
          # Set blur size as a fraction of bounding box size
          blur = int(max(w, h) * blur)  # 10% of bounding box size
          if blur % 2 == 0:  # Ensure blur size is odd
            blur += 1
          masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

        # Convert mask to single channel where pixel values are from the alpha channel of the current mask
        mask = Image.fromarray(masked_diff)

        last_mask = mask  # Update last_mask with the final mask after dilation and feathering

  # Convert numpy arrays to PIL Images
  input1 = Image.fromarray(img)
  input2 = Image.fromarray(original_img)

  # Resize mask to match image size
  #mask = Image.fromarray(mask)
  mask = mask.resize(input1.size)

  # Ensure images are the same size
  assert input1.size == input2.size == mask.size

  # Paste input1 onto input2 using the mask
  input2.paste(input1, (0,0), mask)

  # Convert the final PIL Image back to a numpy array
  input2 = np.array(input2)

  #input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
  cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
  
  return input2, mask

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, results_file='last_detected_face.pkl'):
    # If results file exists, load it and return
    if os.path.exists(results_file):
        print('Using face detection data from last input')
        with open(results_file, 'rb') as f:
            return pickle.load(f)

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    from tqdm import tqdm
    tqdm = partial(tqdm, position=0, leave=True)

    for image, rect in tqdm(zip(images, face_rect(images)), total=len(images), desc="detecting face in every frame", ncols=100):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('未检测到人脸！请确保视频中的每一帧都包含一个人脸')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if str(args.nosmooth) == 'False': boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    # Save results to file
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    return results

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print("\r" + " " * 100, end="\r")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()

        if idx == 555:
            print("error begin ...")

        if idx == len(face_det_results):

            print("视频不足，语音仍未遍历完，请检查此处")
            #删除上次留存的视觉面部识别文件，重新生成一次
            os.remove(f"last_detected_face.pkl")
            if args.box[0] == -1:
                if not args.static:
                    face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
                else:
                    face_det_results = face_detect([frames[0]])
            else:
                print('Using the specified bounding box instead of face detection...')
                y1, y2, x1, x2 = args.box
                face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda'

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def main():
    args.img_size = 96
    frame_number = 11

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        if args.fullres != 1:
          print('Resizing video...')
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))
     
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame) 

    if not args.audio.endswith('.wav'):
        print('Converting audio to .wav')
        subprocess.check_call([
              "ffmpeg", "-y", "-loglevel", "error",
              "-i", args.audio,
              "temp/temp.wav",
          ])
        args.audio = 'temp/temp.wav'
        
    print('analysing audio...')
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
 
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    
    mel_chunks = []

    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1;
    
    full_frames = full_frames[:len(mel_chunks)]
    if str(args.preview_settings) == 'True':
      full_frames = [full_frames[0]]
      mel_chunks = [mel_chunks[0]]
    print (str(len(full_frames))+' frames to process')
    batch_size = args.wav2lip_batch_size
    if str(args.preview_settings) == 'True':
      gen = datagen(full_frames, mel_chunks)
    else:
      gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(
    gen,
    total=int(np.ceil(float(len(mel_chunks))/batch_size)),
    desc="Processing Wav2Lip",ncols=100
)):
        if i == 0:

          if not args.quality=='Fast':
            print(f"mask size: {args.mask_dilation}, feathering: {args.mask_feathering}")  
            if not args.quality=='Improved':   
              print("Loading", args.sr_model)
              run_params = load_sr()

          print("Starting...")
          frame_h, frame_w = full_frames[0].shape[:-1]
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
          out = cv2.VideoWriter('temp/result.mp4', fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to('cuda')
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to('cuda')

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            #cv2.imwrite('temp/f.jpg', f)
            
            y1, y2, x1, x2 = c

            if str(args.debug_mask) == 'True' and args.quality != "Experimental": #makes the background black & white so you can see the mask better
              f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
              f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            of=f
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]

            if args.quality=='Enhanced':
              p = upscale(p, run_params)

            if args.quality in ['Enhanced', 'Improved']:
              if str(args.mouth_tracking) == 'True':
                for i in range(len(frames)):
                  p, last_mask = create_tracked_mask(p, cf)
              else:
                for i in range(len(frames)):
                  p, last_mask = create_mask(p, cf)
		      

            f[y1:y2, x1:x2] = p
            #cv2.imwrite('temp/p.jpg', f)

            if args.quality=='Experimental':
              last_mask = None
              for i in range(len(frames)):
                f, last_mask = Experimental(f, of,run_params)

            if str(args.preview_settings) == 'True':
              cv2.imwrite('temp/preview.jpg', f)

            else:
              out.write(f)

    out.release()
    
    if str(args.preview_settings) == 'False':
      print("converting to final video")

      subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", "temp/result.mp4",
        "-i", args.audio,
        "-c:v", "h264_nvenc",
        args.outfile ,
      ])


model = detector = detector_model = None
def do_load(checkpoint_path):
    global model, detector, detector_model
    model = load_model(checkpoint_path)
    detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
    detector_model = detector.model

def face_rect(images):
  face_batch_size = 8
  num_batches = math.ceil(len(images) / face_batch_size)
  prev_ret = None
  for i in range(num_batches):
      batch = images[i * face_batch_size: (i + 1) * face_batch_size]
      all_faces = detector(batch)  # return faces list of all images
      for faces in all_faces:
          if faces:
              box, landmarks, score = faces[0]
              prev_ret = tuple(map(int, box))
          yield prev_ret

if __name__ == '__main__':
    args = parser.parse_args()
    do_load(args.checkpoint_path)
    main()
