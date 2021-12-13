from PIL import Image
from time import sleep
import pandas as pd

import requests
import argparse
import random
import shutil
import tqdm
import csv
import os

def download_img(dataframe):
    no_img = []

    ''' Downloads images and stores for training. '''
    for i, image_url in enumerate(tqdm.tqdm(dataframe['scaled'])):
        if i >= args.start:
            try:
                filename = args.outdir + '/' + str(list(dataframe.index)[i]) + '.jpg'

                # Open the url image, set stream to True, this will return the stream content
                r = requests.get(image_url, stream = True)
                # sleep(random.uniform(1, 2))
                
                # Check if the image was retrieved successfully
                if r.status_code == 200:
                    
                    # Set decode_content value to True, otherwise the downloaded image file's size will be zero
                    r.raw.decode_content = True

                    # Open a local file with wb permission.
                    with open(filename,'wb') as f:
                        shutil.copyfileobj(r.raw, f)

                    # print('Image sucessfully Downloaded: ',filename)
                else:
                    no_img.append(filename)
            except:
                sleep(10)
                continue

        if i >= args.stop:
            break


def download_missing(dataframe, outdir):
    files = os.listdir(outdir)
    files = map(lambda each: each.strip(".jpg"), files)
    files = sorted(list(map(int, files)))
    
    missing = list(set(range(max(files) + 1)) - set(files))
    

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Download images.')

    parser.add_argument('--data_path', dest='datapath', required=True, help='Path to dataset')
    parser.add_argument('--out_dir', dest='outdir', required=True, help='Save images in this directory')
    parser.add_argument('--missing', dest='missing', action='store_true', help='Download missing images')
    parser.add_argument('--start_from', type=int, dest='start', help='Start from left off point')
    parser.add_argument('--stop_at', type=int, dest='stop', help='Stop at certain point')

    args = parser.parse_args()

    if not os.path.isfile(args.datapath):
        raise FileExistsError('Training dataset does not exist.')
    
    df = pd.read_csv(args.datapath)

    if args.missing:
        download_missing(df, args.outdir)
