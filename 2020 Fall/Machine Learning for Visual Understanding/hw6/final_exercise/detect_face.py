# Lint as: python3
# Joon Son Chung, adapted from https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py

import argparse
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', default="ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output', default="out.jpg",
                      help='File path for the result image with annotations')
  args = parser.parse_args()

  ## ========== ========== ===========
  ## Load the network
  ## ========== ========== ===========

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  ## ========== ========== ===========
  ## Compute bounding boxes
  ## ========== ========== ===========

  image = Image.open(args.input)
  _, scale = common.set_resized_input(
      interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  objs = detect.get_objects(interpreter, args.threshold, scale)
  print('%.2f ms' % (inference_time * 1000))

  ## ========== ========== ===========
  ## Crop the image
  ## ========== ========== ===========

  ## Ensure that there is only one face in the image
  assert len(objs) == 1

  bbox = objs[0].bbox

  sx = int((bbox[0]+bbox[2])/2) 
  sy = int((bbox[1]+bbox[3])/2) 
  ss = int(max((bbox[3]-bbox[1]),(bbox[2]-bbox[0]))/2.5)

  print((sx-ss, sy-ss, sx+ss, sy+ss))
  
  cropped_image = image.crop((sx-ss, sy-ss, sx+ss, sy+ss))
  cropped_image.resize((240, 240)).save(args.output)

if __name__ == '__main__':
  main()