

from PIL import Image, ImageDraw
img = Image.open('dataset/chinese_books/微信图片_20200719140117.jpg').convert('RGB')
shape = img.size

draw = ImageDraw.Draw(img)
with open('output/res_微信图片_20200719140117.txt', 'r', encoding='utf-8') as fp:
    for i, line in enumerate(fp):
        line = line.strip()
        if not line:
            continue
        rec = [int(num) for num in line.split(',')]
        draw.rectangle(rec, outline=(255, 0, 0), width=3)
        img.save('output/%d.jpg' % i)
