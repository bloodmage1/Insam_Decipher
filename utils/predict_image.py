import torch
from PIL import Image, ImageDraw, ImageFont
import cv2

def predict_image(model, transform, image_file_rgb):
    image = transform(image_file_rgb).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label_mapping = {0: '상', 1: '중', 2: '하', 3: '최하'}
        return label_mapping[predicted.item()]

def annotate_image(image, label):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("malgun.ttf", 150)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])

    margin = 200
    image_width, image_height = image.size
    box = [
        image_width - text_size[0] - 2 * margin,
        margin,
        image_width - margin,
        text_size[1] + 2 * margin
    ]

    draw.rectangle(box, outline="red", width=30)  # 빨간 사각형 그리기
    draw.text((box[0] + margin-100, box[1] + margin-130), label, fill="red", font=font)  # 텍스트 삽입

    stamp_cv = cv2.imread("./app_img/stamp.png", cv2.IMREAD_UNCHANGED)  #
    stamp_cv_resized = cv2.resize(stamp_cv, (300, 300))  # 크기 조정

    stamp_pil = Image.fromarray(cv2.cvtColor(stamp_cv_resized, cv2.COLOR_BGRA2RGBA))

    # 스탬프 위치 계산
    stamp_x = box[2] - 80  # 사각형의 오른쪽으로 100픽셀
    stamp_y = box[1]  -50 # 사각형의 위로 100픽셀

    # 스탬프 이미지 삽입
    image.paste(stamp_pil, (stamp_x, stamp_y), stamp_pil)

    # 이미지 회전
    image = image.rotate(-5, expand=1)  # 이미지를 오른쪽으로 10도 회전

    return image  # 주석이 추가된 이미지 반환