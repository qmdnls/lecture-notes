# Lint as: python3
# Joon Son Chung
r"""
The input to this code should be the cropped images.
Note that the reference images are already cropped.
"""

import argparse
import time
import numpy
import pdb
import os
import glob

from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from sklearn import metrics

def numpy_loader(filename):
    image = Image.open(filename)
    image = image.resize((256,256),resample=Image.BILINEAR)
    image = image.crop((16,16,240,240))
    image = numpy.asarray(image, dtype=numpy.float32) / 255.
    image = numpy.subtract(image, numpy.array([0.485,0.456,0.406]))
    image = numpy.divide(image, numpy.array([0.229,0.224,0.225]))
    image = numpy.expand_dims(image, 0) 
    return image

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    scale, zero_point = input_details['quantization']
    input_tensor[:, :] = numpy.uint8(input / scale + zero_point)

def cos_sim(com_feat,ref_feat):
    """ Takes as input two one-dimensional numpy arrays and computes their cosine similarity

    Args:
        com_feat (1-d numpy array): computed feature embedding vector
        ref_feat (1-d numpy array): reference feature embedding vector
    """
    # Fill this in
    a = numpy.squeeze(com_feat)
    b = numpy.squeeze(ref_feat)
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

def main():

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default="my_model_edgetpu.tflite",
                                            help='File path of .tflite file.')
    parser.add_argument('-d', '--dataset_path', required=True,
                                            help='Image to the dataset.')
    parser.add_argument('-i', '--input', required=True,
                                            help='Image to the dataset.')
    args = parser.parse_args()

    ## ========== ========== ===========
    ## Load the network
    ## ========== ========== ===========

    # Fill this in
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    ## ========== ========== ===========
    ## Compute embeddings
    ## ========== ========== ===========
    
    # Fill this in
   
    files = glob.glob(args.dataset_path+'/*.jpg')

    embeddings = {} # File name -> embedding dict

    print("Computing embeddings...")

    for i, file in enumerate(files):
        # Print progress since this can take a moment
        print(i, "of", len(files), "images processed", end="\r", flush=True)

        # Load the image
        image = numpy_loader(file)
        set_input_tensor(interpreter, image)
        
        # Run tensor through the model
        interpreter.invoke()
        
        # Obtain output tensor, de-quantize and convert to numpy
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details['index'])
        scale, zero_point = output_details['quantization']
        output = numpy.array(output, dtype=numpy.float32)
        output = scale * (output - zero_point)
        
        # Save embedding in our embedding dict
        embeddings[file] = output

    # Flush the status line
    print("", end="\r", flush=True)

    ## ========== ========== ===========
    ## Compute embedding of the input
    ## ========== ========== ===========

    # Fill this in

    # Load reference image
    ref = numpy_loader(args.input) 
    
    # Similarly compute the embedding
    set_input_tensor(interpreter, ref)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details['index'])
   
    # Convert to numpy float32 and de-quantize
    scale, zero_point = output_details['quantization']
    output = numpy.array(output, dtype=numpy.float32)
    emb_ref = scale * (output - zero_point)

    ## ========== ========== ===========
    ## Compute cosine distances
    ## ========== ========== ===========

    # Fill this in
    
    # Compute cosine similarity scores
    scores = {filename: cos_sim(emb, emb_ref) for filename, emb in embeddings.items()}
    
    # Get the file with the highest cosine similarity
    lookalike = max(scores, key=scores.get)

    # Compute and print the cosine similarity
    print("Lookalike:", lookalike)
    print("Cosine similarity:", scores[lookalike]) 

if __name__ == '__main__':
    main()
