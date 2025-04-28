import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import os
from PIL import Image
import requests
import zipfile
import tarfile
import glob
import random

# mps 설정 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DIV2K 데이터셋 다운로드 및 준비
class DIV2KDataset(Dataset):
    def __init__(self, root_dir, scale=4, train=True, image_size=256, download=True):
        """
        DIV2K 데이터셋 로더
        Args:
            root_dir: 데이터셋이 저장될 경로
            scale: 초해상도 스케일 (기본값: 4x)
            train: 학습 또는 검증 데이터셋 선택
            image_size: 고해상도 이미지 크기
            download: 데이터셋 다운로드 여부
        """
        self.root_dir = root_dir
        self.scale = scale
        self.train = train
        self.image_size = image_size
        self.lr_size = image_size // scale
        
        # 폴더 생성
        os.makedirs(root_dir, exist_ok=True)
        
        # 학습/검증 데이터셋 경로 설정
        self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR' if train else 'DIV2K_valid_HR')
        self.lr_dir = os.path.join(root_dir, f'DIV2K_train_LR_bicubic/X{scale}' if train else f'DIV2K_valid_LR_bicubic/X{scale}')
        
        # 데이터셋 다운로드
        if download:
            self._download_div2k()
        
        # 이미지 파일 경로 가져오기
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, '*.png')))
        
        # 변환 정의
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def _download_div2k(self):
        """DIV2K 데이터셋 다운로드 및 압축 해제"""
        if self.train:
            # 학습 데이터셋 다운로드
            if not os.path.exists(self.hr_dir):
                hr_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
                hr_file = os.path.join(self.root_dir, "DIV2K_train_HR.zip")
                
                print("DIV2K 학습 HR 이미지 다운로드 중...")
                self._download_file(hr_url, hr_file)
                
                print("DIV2K 학습 HR 이미지 압축 해제 중...")
                with zipfile.ZipFile(hr_file, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir)
            
            # LR 이미지 다운로드
            lr_dir_parent = os.path.dirname(self.lr_dir)
            if not os.path.exists(lr_dir_parent):
                lr_url = f"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X{self.scale}.zip"
                lr_file = os.path.join(self.root_dir, f"DIV2K_train_LR_bicubic_X{self.scale}.zip")
                
                print(f"DIV2K 학습 LR X{self.scale} 이미지 다운로드 중...")
                self._download_file(lr_url, lr_file)
                
                print(f"DIV2K 학습 LR X{self.scale} 이미지 압축 해제 중...")
                with zipfile.ZipFile(lr_file, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir)
        else:
            # 검증 데이터셋 다운로드
            if not os.path.exists(self.hr_dir):
                hr_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
                hr_file = os.path.join(self.root_dir, "DIV2K_valid_HR.zip")
                
                print("DIV2K 검증 HR 이미지 다운로드 중...")
                self._download_file(hr_url, hr_file)
                
                print("DIV2K 검증 HR 이미지 압축 해제 중...")
                with zipfile.ZipFile(hr_file, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir)
            
            # LR 이미지 다운로드
            lr_dir_parent = os.path.dirname(self.lr_dir)
            if not os.path.exists(lr_dir_parent):
                lr_url = f"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{self.scale}.zip"
                lr_file = os.path.join(self.root_dir, f"DIV2K_valid_LR_bicubic_X{self.scale}.zip")
                
                print(f"DIV2K 검증 LR X{self.scale} 이미지 다운로드 중...")
                self._download_file(lr_url, lr_file)
                
                print(f"DIV2K 검증 LR X{self.scale} 이미지 압축 해제 중...")
                with zipfile.ZipFile(lr_file, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir)
    
    def _download_file(self, url, dest):
        """파일 다운로드 헬퍼 함수"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(dest, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        
        progress_bar.close()
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_img_path = self.hr_files[idx]
        
        # 고해상도 이미지 로드 및 처리
        hr_img = Image.open(hr_img_path).convert('RGB')
        
        # 크롭할 영역 랜덤 선택 (고정된 크기로)
        width, height = hr_img.size
        if width < self.image_size or height < self.image_size:
            # 이미지가 너무 작으면 패딩 추가
            padded = Image.new('RGB', (max(width, self.image_size), max(height, self.image_size)), (0, 0, 0))
            padded.paste(hr_img, (0, 0))
            hr_img = padded
            width, height = hr_img.size
            
        left = random.randint(0, width - self.image_size)
        top = random.randint(0, height - self.image_size)
        hr_patch = hr_img.crop((left, top, left + self.image_size, top + self.image_size))
        
        # 저해상도 패치 생성 (직접 다운샘플링)
        lr_patch = hr_patch.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        
        # 텐서 변환
        hr_tensor = self.hr_transform(hr_patch)
        lr_tensor = self.lr_transform(lr_patch)
        
        return {'lr': lr_tensor, 'hr': hr_tensor}

# 향상된 UNet 모델 (고해상도 이미지 처리용)
class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64, n_blocks=8):
        super().__init__()
        
        # 시간 임베딩
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 초기 컨볼루션
        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # 다운샘플링 경로
        self.down_blocks = nn.ModuleList()
        ch = hidden_channels
        for i in range(3):  # 3단계 다운샘플링
            down_block = nn.Sequential(
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, ch * 2),
                nn.SiLU(),
                nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, ch * 2),
                nn.SiLU()
            )
            self.down_blocks.append(down_block)
            ch *= 2
        
        # 중간 레즈넷 블록
        self.mid_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.mid_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, ch),
                    nn.SiLU(),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, ch),
                    nn.SiLU()
                )
            )
        
        # 업샘플링 경로
        self.up_blocks = nn.ModuleList()
        for i in range(3):  # 3단계 업샘플링
            up_block = nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, ch // 2),
                nn.SiLU(),
                nn.Conv2d(ch // 2, ch // 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, ch // 2),
                nn.SiLU()
            )
            self.up_blocks.append(up_block)
            ch //= 2
        
        # 최종 출력 컨볼루션
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t):
        # 시간 임베딩
        t = t.float().unsqueeze(-1) / 500  # 시간 정규화
        t = self.time_mlp(t)
        
        # 초기 특징 추출
        x = self.init_conv(x)
        h = x
        
        # 다운샘플링
        features = [h]
        for down_block in self.down_blocks:
            h = down_block(h)
            features.append(h)
        
        # 중간 레즈넷 블록
        for mid_block in self.mid_blocks:
            h = h + mid_block(h)  # 레즈넷 연결
        
        # 업샘플링 (스킵 연결 없이)
        for up_block in self.up_blocks:
            h = up_block(h)
        
        # 최종 출력
        return self.out_conv(h)

# 확산 모델
class DiffusionModel(nn.Module):
    def __init__(self, model, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        
        # 베타 스케줄
        self.betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 샘플링을 위한 계수
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """주어진 시작점 x_start와 타임스텝 t에서 노이즈 추가"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 인덱싱을 위한 조정
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def forward(self, x_start):
        """학습을 위한 forward pass"""
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy, _ = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, t)
        
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """샘플링 과정의 단일 스텝"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        
        # 모델 예측
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        # t > 0일 때만 노이즈 추가
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            model_mean = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return model_mean
    
    @torch.no_grad()
    def sample(self, batch_size=4, img_size=256, channels=3):
        """완전한 역확산 과정을 통한 샘플 생성"""
        shape = (batch_size, channels, img_size, img_size)
        x = torch.randn(shape, device=device)
        
        # 역확산 과정
        for t in tqdm(reversed(range(0, self.n_steps)), total=self.n_steps):
            x = self.p_sample(x, torch.full((batch_size,), t, device=device, dtype=torch.long))
            
        # [-1, 1] 범위로 맞추기
        x = torch.clamp(x, -1, 1)
        # [0, 1] 범위로 변환
        x = (x + 1) / 2
        
        return x

# 초해상도 확산 모델
class SRDiffusion(nn.Module):
    def __init__(self, image_size=256, hidden_channels=64, n_steps=1000):
        super().__init__()
        self.image_size = image_size
        
        # 향상된 UNet 모델
        self.model = EnhancedUNet(
            in_channels=3,
            out_channels=3, 
            hidden_channels=hidden_channels
        )
        
        # 확산 모델
        self.diffusion = DiffusionModel(
            model=self.model,
            n_steps=n_steps
        )
    
    def forward(self, x):
        """학습 손실 계산"""
        return self.diffusion(x)
    
    def sample(self, batch_size=4):
        """샘플 생성"""
        return self.diffusion.sample(batch_size, self.image_size, 3)

# 고정된 크기의 패치를 생성하는 사용자 정의 collate 함수
def custom_collate(batch):
    lr_list = [item['lr'] for item in batch]
    hr_list = [item['hr'] for item in batch]
    
    # 모든 텐서가 같은 크기인지 확인
    lr_sizes = set([img.size() for img in lr_list])
    hr_sizes = set([img.size() for img in hr_list])
    
    if len(lr_sizes) > 1 or len(hr_sizes) > 1:
        # 크기가 다르면 동일한 크기로 조정
        target_lr_size = lr_list[0].size()
        target_hr_size = hr_list[0].size()
        
        for i in range(len(lr_list)):
            if lr_list[i].size() != target_lr_size:
                lr_list[i] = F.interpolate(lr_list[i].unsqueeze(0), size=target_lr_size[1:], 
                                           mode='bilinear', align_corners=False).squeeze(0)
            if hr_list[i].size() != target_hr_size:
                hr_list[i] = F.interpolate(hr_list[i].unsqueeze(0), size=target_hr_size[1:], 
                                           mode='bilinear', align_corners=False).squeeze(0)
    
    # 배치로 묶기
    lr_batch = torch.stack(lr_list)
    hr_batch = torch.stack(hr_list)
    
    return {'lr': lr_batch, 'hr': hr_batch}

# 학습 함수 (데이터셋에 맞게 수정)
def train(model, train_loader, optimizer, epochs, device, save_interval=5):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            hr_images = batch['hr'].to(device)
            
            optimizer.zero_grad()
            
            # 학습 (고해상도 이미지로 학습)
            loss = model(hr_images)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 주기적으로 모델 저장
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"sr_diffusion_model_epoch_{epoch+1}.pth")
            
            # 샘플 생성
            generate_samples(model, epoch)

# 샘플 생성 및 시각화
def generate_samples(model, epoch, num_samples=4):
    model.eval()
    with torch.no_grad():
        # 샘플 생성
        samples = model.sample(batch_size=num_samples)
        
        # 시각화
        samples_cpu = samples.cpu()
        
        plt.figure(figsize=(20, 20))
        for i in range(num_samples):
            plt.subplot(2, 2, i+1)
            plt.imshow(samples_cpu[i].permute(1, 2, 0))
            plt.axis('off')
        
        os.makedirs('sr_samples', exist_ok=True)
        plt.savefig(f"sr_samples/sample_epoch_{epoch+1}.png")
        plt.close()

# 메인 함수
def main():
    # 하이퍼파라미터
    image_size = 256  # 고해상도 이미지 크기
    batch_size = 2     # 더 큰 이미지를 사용하므로 배치 크기 감소
    epochs = 50
    hidden_channels = 64
    lr = 1e-4
    scale = 4  # 초해상도 배율
    
    # DIV2K 데이터셋 준비
    train_dataset = DIV2KDataset(
        root_dir='./div2k_data',
        scale=scale,
        train=True,
        image_size=image_size,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate,  # 사용자 정의 collate 함수 사용
        drop_last=True  # 불완전한 배치 제거
    )
    
    # 모델 초기화
    model = SRDiffusion(
        image_size=image_size,
        hidden_channels=hidden_channels,
        n_steps=1000
    ).to(device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 학습 시작
    train(model, train_loader, optimizer, epochs, device)
    
    # 최종 모델 저장
    torch.save(model.state_dict(), "sr_diffusion_final_model.pth")
    
if __name__ == "__main__":
    main()