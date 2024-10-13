import torch
from torchvision import models, transforms
from PIL import Image

# ResNet18 모델 불러오기
model = models.resnet18(pretrained=True)
model.eval()  # 평가 모드

# 이미지 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 파일 로드
#image_path = '/Users/hyeonggeun_kim/Desktop/noised_tumbler.jpg'
image_path = '/Users/hyeonggeun_kim/Downloads/IMG_9861.jpg'
# 이미지 파일 로드 및 RGBA를 RGB로 변환
img = Image.open(image_path)

if img.mode == 'RGBA':
    img = img.convert('RGB')

# 이후에 이미지 전처리 적용
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# 예측 수행
with torch.no_grad():
    out = model(batch_t)

# 가장 높은 확률의 클래스 가져오기
_, index = torch.max(out, 1)

# 클래스 레이블을 읽기 위해 ImageNet 클래스 인덱스 불러오기
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

import json
import urllib.request

response = urllib.request.urlopen(LABELS_URL)
labels = json.load(response)

# 예측 결과 출력
predicted_label = labels[index.item()]
print(f"예측된 클래스: {predicted_label}")
