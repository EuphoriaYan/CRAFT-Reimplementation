from PIL import Image, ImageDraw
import argparse
import xml.sax
import json


class PageHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.text = ""
        self.type = ""
        self.bbox = (0, 0, 0, 0)

    # 元素开始调用
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "object":
            print("Start of Element")
            text = attributes["name"]
            print("Text:", text)

    # 元素结束调用
    def endElement(self, tag):
        if self.CurrentData == "type":
            print("Type:", self.type)
        elif self.CurrentData == "format":
            print("Format:", self.format)
        elif self.CurrentData == "year":
            print("Year:", self.year)
        elif self.CurrentData == "rating":
            print("Rating:", self.rating)
        elif self.CurrentData == "stars":
            print("Stars:", self.stars)
        elif self.CurrentData == "description":
            print("Description:", self.description)
        self.CurrentData = ""

    # 读取字符时调用
    def characters(self, content):
        if self.CurrentData == "name":
            self.text = content[1:]
            type = content[0]
            try:
                type = int(type)
            except ValueError as e:
                print(e)
            assert type in [0, 1, 2]
            if type == 0:
                self.type = 'S'
            if type == 1:
                self.type = 'L'
            if type == 2:
                self.type = 'R'
        elif self.CurrentData == "bndbox":
            self.bbox = (content.xmin, content.ymin, content.xmax, content.ymax)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', type=str, required=True)
    parser.add_argument('--input_json', type=str, default=None)
    parser.add_argument('--input_xml', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    img = Image.open(args.input_img).convert('RGB')
    draw = ImageDraw.Draw(img)
    if args.input_json is not None and args.input_xml is not None:
        raise ValueError
    if args.input_json is None and args.input_xml is None:
        raise ValueError
    if args.input_json is not None:
        json_file = args.input_json
        json_data = json.load(open(json_file, 'r', encoding='utf-8'))
        regions = json_data['regions']
        for idx, region in enumerate(regions):
            bbox = region['boundingBox']
            bbox = (bbox['left'], bbox['top'], bbox['left']+bbox['width'], bbox['top']+bbox['height'])
            draw.rectangle(bbox, outline=(255,0,0), width=3)
            draw.text((bbox[0]+5,bbox[1]+5), str(idx), fill=(0,0,255))
        img.show()
        # img.save('output/gt_000002.jpg')
    if args.input_xml is not None:
        xml_file = args.input_xml
        xml_parser = xml.sax.make_parser()
        xml_parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        handler = PageHandler()
        xml_parser.setContentHandler(handler)
        xml_parser.parse(xml_file)
