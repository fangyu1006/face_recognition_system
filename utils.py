import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def draw_txt_image(image, strs, local, sizes, colour):
    print(strs)
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font_path = "./simsun.ttc"
    font = ImageFont.truetype(font_path,sizes, encoding="utf-8")
    #font = ImageFont.truetype(font_path,sizes)
    draw.text(local, unicode(strs), colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image
