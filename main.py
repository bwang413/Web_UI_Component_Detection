import ultralytics
from ultralyticsplus import YOLO, render_result
import os, torch
from PIL import Image


def load_image(folder_path = './image'):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    images = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        images.append(image)

    return images


if __name__ == '__main__':
    images = load_image()
    model = YOLO('form_detection_customized.pt')

    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    image = './image/1.png'
    results = model.predict(image)
    render = render_result(model=model, image=image, result=results[0])
    render.show()
