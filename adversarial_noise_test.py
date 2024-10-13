from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# 모델 로드: ResNet18 사전 학습된 모델 사용
model = models.resnet18(pretrained=True)
model.eval()  # 평가 모드로 설정

# FGSM 공격 함수 정의
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def load_image(image_path):
    img = Image.open(image_path)
    
    # RGBA 이미지라면 RGB로 변환
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가
    return img_tensor

def create_adversarial_image(image_tensor, epsilon=0.1):
    image_tensor.requires_grad = True
    
    # 모델 예측
    output = model(image_tensor)
    init_pred = output.max(1, keepdim=True)[1]
    
    # 타겟을 1D 텐서로 변환
    target = init_pred.view(-1)  # 1D 형태로 변경
    
    # 손실 계산
    loss = nn.CrossEntropyLoss()(output, target)
    
    # 역전파로 기울기 계산
    model.zero_grad()
    loss.backward()
    
    # 기울기 추출
    data_grad = image_tensor.grad.data
    
    # 적대적 이미지 생성
    perturbed_image = fgsm_attack(image_tensor, epsilon, data_grad)
    
    return image_tensor, perturbed_image


# 원본 이미지 로드
image_path = '/Users/hyeonggeun_kim/Desktop/tumbler.jpg'
original_image = load_image(image_path)

# 적대적 이미지 생성
original_image, adversarial_image = create_adversarial_image(original_image)

# 결과 시각화
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())  # detach() 추가

plt.subplot(1, 2, 2)
plt.title("Adversarial Image")
plt.imshow(adversarial_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())  # detach() 추가

plt.show()
