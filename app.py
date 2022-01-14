import colorsys
from PIL import Image,ImageStat
import numpy as np
import pandas as pd
import math
import scipy.misc
import imageio
from colorthief import ColorThief
from collections import namedtuple
import glob
import logging
import os
import csv
import cv2

image_list_csv = '/home/saber/workspace/data/sampleimagelist.csv'
source_dir = '/home/saber/workspace/data/gofundme-1.4/all_img/'
outputcsv = 'image-attributes-80k-withbluriness.csv'
'''
yields each file as a tuple of: 1-name of the image_list_csv without extension 2-project_id 3-image_id
'''
def getFiles136k(): 
    df = pd.read_csv(image_list_csv, header=None, skiprows=0) #, nrows=2
    for row in df.itertuples():
        yield (row[2].lower() + '/' + str(row[1])[:3] + '/' + str(row[1]) + '/' + str(row[4]),row[1],row[4])

def getFiles80k():
    source_dir = '/mnt/home/saber/workspace/faceapi/img80k/'
    #backup_dir = '/mnt/home/saber/workspace/faceapi/img80k_processed/'
    for _, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    #os.rename(source_dir + file, backup_dir + file) #remove the file from current directory\
                    yield (source_dir + file,file.split("_")[0],file)


def main():
    #Get image width & height in pixels
    # [xs, ys] = img_file.size
    # max_intensity = 100
    # hues = {}
    #Examine each pixel in the image file
    # for x in range(0, xs):
    #     for y in range(0, ys):
    #         [r, g, b] = img[x, y] #Get the RGB color of the pixel
    #         #Normalize pixel color values
    #         r /= 255.0
    #         g /= 255.0
    #         b /= 255.0
    #         print(r,g,b)

    #         # Convert RGB color to HSV (hue, saturation, value)
    #         [h, s, v] = colorsys.rgb_to_hsv(r, g, b)
    #         print(h,s,v)

    #Brightness
    #img = Image.open(img_file)
    RGB = namedtuple('RGB', 'r g b')
    Attributes = namedtuple('Attributes', 'dominantcolor mean brightness sum median stddev blur')
    for f in getFiles80k():
        path = f[0]
        project_id = f[1].split(".")[0]
        image_id = f[2]
        
        files = glob.glob(path if path.lower().endswith(('.png', '.jpg', '.jpeg')) else source_dir + path + ".*") 
        if files:
            file = files[0]
            try:
                img_file = Image.open(file)
                #img = img_file.load()
                dc = getdominantcolor(img_file)
       
                dominantcolor = RGB(round(dc[0]),round(dc[1]),round(dc[2]))
                #faces[dominantcolor] = dominantcolor
                #print("dominantcolor", dominantcolor)

                stat = ImageStat.Stat(img_file)
                m = stat.mean
                mean = RGB(round(m[0]),round(m[1]),round(m[2]))
                #faces[mean] = mean
                #print("mean", mean)

                brightness = getbrightness(mean.r,mean.g,mean.b)
                #faces[brightness] = brightness
                #print("brightness", brightness)

                m = stat.sum
                sum = RGB(round(m[0]),round(m[1]),round(m[2]))
                #faces[sum] = sum
                #print("sum", sum)

                m = stat.median
                median = RGB(round(m[0]),round(m[1]),round(m[2]))
                #faces[median] = median
                #print("median", median)

                m = stat.stddev
                stddev = RGB(round(m[0]),round(m[1]),round(m[2]))
                #faces[stddev] = stddev
                #print("stddev", stddev)

                blur = getbluriness(file)

                save_csv(project_id,file,Attributes(dominantcolor, mean, brightness, sum, median, stddev, blur))
            except:
                logging.exception(path + "," + project_id + "," + image_id)
    

def getdominantcolor(image):
    # color_thief = ColorThief(img_path)
    # dominant_color = color_thief.get_color(quality=1)
    # return dominant_color
    #Resizing parameters
    width, height = 150,150
    #image = Image.open(filename)
    image = image.resize((width, height),resample = 0)
    #Get colors from image object
    pixels = image.getcolors(width * height)
    #Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    #Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    if isinstance(dominant_color, int):
        return None
    return dominant_color    

def getbrightness(r,g,b):
    brightness = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
    return brightness

def getbluriness(imagePath):
    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return str(round(fm,2)) #bluriness, let's assume that threshold = 100, so any fm < 100 is blurry

def save_csv(project_id,file,attr):
    file_exists = os.path.isfile(outputcsv)
    f = open(outputcsv, 'a')
    with f:
        fnames = ['project_id','image_id','mean_r','mean_g','mean_b','brightness','sum_r','sum_g','sum_b','median_r','median_g','median_b', 'stddev_r','stddev_g','stddev_b', 'dominantcolor_r','dominantcolor_g','dominantcolor_b','blur']
        writer = csv.DictWriter(f, fieldnames=fnames)
        if not file_exists:
            writer.writeheader()
        row = {
            'project_id': project_id,
            'image_id': file[len(source_dir):],
            'mean_r': attr.mean.r,
            'mean_g': attr.mean.g,
            'mean_b': attr.mean.b,
            'brightness': attr.brightness,
            'sum_r': attr.sum.r,
            'sum_g': attr.sum.g,
            'sum_b': attr.sum.b,
            'median_r': attr.median.r,
            'median_g': attr.median.g,
            'median_b': attr.median.b,
            'stddev_r': attr.stddev.r,
            'stddev_g': attr.stddev.g,
            'stddev_b': attr.stddev.b,
            'dominantcolor_r': attr.dominantcolor.r,
            'dominantcolor_g': attr.dominantcolor.g,
            'dominantcolor_b': attr.dominantcolor.b,
            'blur': attr.blur,
        }
        writer.writerow(row)

if __name__ == "__main__":
    main()


#resources:
#RGB Image Analysis http://marksolters.com/programming/2015/02/27/rgb-histograph.html
#Hue https://en.wikipedia.org/wiki/Hue
#HSV https://en.wikipedia.org/wiki/HSL_and_HSV
#Average and Dominant colors of an image https://stackoverflow.com/a/43111221/87088
#Brightness https://stackoverflow.com/a/3498247/87088
# Pillow imagestat: https://pillow.readthedocs.io/en/4.1.x/reference/ImageStat.html
# Interests:
## dominant color, average color, hue, saturation, value, brightness, colorfulness, 
