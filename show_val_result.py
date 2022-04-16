# -*- coding:utf-8 -*-
"""
Date     : 2022-03-26
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : show_val_result.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def show_result():
    confidence = 0.75
    data_dir = "val_data_predict.txt"

    with open(data_dir, 'r') as f:
        datas = f.readlines()
    with open("./train/val.txt", 'r') as f:
        datax = f.readlines()
    for idx, line in enumerate(datax):
        img_dir = line.split()[0].replace('\n', '')
        data = datas[idx].split('|')
        bbox1 = []
        bbox2 = []
        if data[0] != 'x':
            for i in data[0].split(' '):
                bbox1.append(list(map(float, i.split(','))))
        if data[1] != '\n':
            for i in data[1].split(' '):
                bbox2.append(list(map(int, i.split(',')[:-1])))

        image = Image.open(img_dir)
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = 1
        # transform = A.Compose([A.CLAHE(p=1.), A.MedianBlur(blur_limit=3, p=1.)])  # A.CLAHE(p=1.)
        # transformed = transform(image=np.array(image))
        # image = Image.fromarray(transformed['image'])

        draw = ImageDraw.Draw(image)
        for bbox in bbox1:
            if bbox[-1] >= confidence:
                box = list(map(int, bbox[:-1]))
                score = bbox[-1]

                left, top, right, bottom = box

                label = '{:.2f}'.format(score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for j in range(thickness):
                    draw.rectangle([left + j, top + j, right - j, bottom - j], outline='red')
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='red')
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        for bbox in bbox2:
            draw.rectangle(bbox, outline="yellow")
        del draw
        # image.show()
        # if len(bbox1) != 0:
        #     image.save('./tmp/' + str(idx).zfill(6) + '.jpg')
        image.save('./tmp/' + str(idx).zfill(6) + '.jpg')
        frame = cv2.cvtColor(np.array(image.resize((400, 400))), cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


def show_raw():
    data_dir = "./train/val.txt"
    with open(data_dir, 'r') as f:
        datas = f.readlines()
    for data in datas:
        bbox1 = []
        image_raw = Image.open(data.split(' ')[0].replace('\n', ''))
        if len(data.split(' ')) != 1:
            for gt in data.split(' ')[1:]:
                bbox1.append(list(map(int, gt.split(',')[:-1])))
        draw = ImageDraw.Draw(image_raw)
        for bbox in bbox1:
            draw.rectangle(bbox, outline="red")
        del draw
        image_raw.show()
        _ = input()


if __name__ == '__main__':
    show_result()
