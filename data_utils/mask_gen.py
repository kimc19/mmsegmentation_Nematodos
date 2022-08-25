#Generador de máscaras.
# 
# Este script utiliza un archivo .json con landmarks de nematodos para convertirlos en máscaras para segmentación semántica.
# 
# Autor: Marco Zolla
# 
# Editado: Kimberly Carvajal


#Libraries
import argparse
import json
import os
import os.path as osp
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser(description='Generador de máscaras a partir de anotaciones VIA.')

parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--label_file', type=str, help='Dirección del file JSON con las anotaciones.')
parser.add_argument('--mask_out', type=str, help='Dirección de la carpeta donde guardar las máscaras.')

args = parser.parse_args()


#Extract image dimensions
def get_img_dims(img_name):
    path = args.img_in + '/' + img_name
    im = Image.open( path )
    return im.size


with open(args.label_file) as f:
    annotations = json.load(f)

img_data = annotations['_via_img_metadata']
img_keys = img_data.keys()
img_filenames=os.listdir(args.img_in)

for i in img_keys:
    current_img = img_data[i]
    img_name = current_img['filename']
    
    if img_name in img_filenames:
        img_dims = get_img_dims(img_name)
        # make a list of touples
        if current_img['regions']:
            x_list = current_img['regions'][0]['shape_attributes']['all_points_x']
            y_list = current_img['regions'][0]['shape_attributes']['all_points_y']
            coords = list(zip(x_list, y_list))
            new_img = Image.new("RGB", img_dims)
            img1 = ImageDraw.Draw(new_img)
            img1.polygon(coords, fill = 'white', outline = 'white')
            
            # save mask
            new_name = osp.splitext(img_name)[0] + '.png'
            new_img = new_img.save(args.mask_out + '/' + new_name )
            print(f'Saved {img_name}\'s mask as {new_name} ')
            
        else:
            print(f'No annotation data for {img_name}. Skipping...')


