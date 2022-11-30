import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path
import glob



def resize_to_heightbywidth(im_tf, mode: str, height:int, width:int) -> tf.image:
    '''
    inputs: im_if, type: tensorflow-tensor
            mode, type: string
    outputs: im_tf, type: tensorflow-tensor        
    '''
    height_old, width_old , _ = im_tf.shape
    if height_old < height or width_old < width :
        print('one of your pictures is smaller than the wanted height, please use the padding option, Padding option is used therefore')
        mode = 'pad'
    
    if mode == 'resize':
        im_tf = tf.image.resize(im_tf,(height,width))
        
    elif mode == 'pad':
        im_tf = tf.image.resize_with_pad(
            im_tf,
            height,width,     
               )
      
    return im_tf




def resize_whole_folder( height :int, width :int, mode :str, images_folder_path):
    """This function loops over a folder with images and resizes it acording to the parameters width and height. 
        If the 'resize' option is used, but the picture is too small, it will be padded automatically to ensure the wandted size.
        

    Args:
        height (int): wanted output
        width (int): wanted output
        mode (str): define either:  'resize' or 'pad 
        images_folder_path (_type_): define the folder of your current pictures width *.png in the end  (see below)
    """

    image_list_resized = []
    for filename in glob.glob(str(images_folder_path)): #assuming gif
        im=Image.open(filename)
        im_tf = img_to_array(im).astype('uint8')  
        im_tf = tf.convert_to_tensor(im_tf)
        resized_image = resize_to_heightbywidth(im_tf, mode, height,width)
        image_list_resized.append(resized_image)
        #print(resized_image)



# Define your path to the image folder
images_folder_path = Path('Group04','notebooks','images', '*.png')
resize_whole_folder(224,224, 'resize', images_folder_path)



