import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from PIL import Image 
from torchvision import transforms  
import numpy as np

class Diffusion(nn.Module):          
    def __init__(self, start, end, timesteps):                  
        super().__init__()       
        
        self.timesteps =  timesteps           
        self.beta = torch.linspace(start, end, timesteps)         
        self.alpha = 1 - self.beta         
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)         
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)         
        self.one_minus_sqrt_alpha_bar = torch.sqrt(1 - self.alpha_bar)              

    def forward(self, x, t):         
        x = x.unsqueeze(0)         
        batch_size = x.size(0)                  
        noise = torch.randn_like(x)         
        noise_img = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1) * x + self.one_minus_sqrt_alpha_bar[t].view(-1, 1, 1, 1) * noise         
        return noise, noise_img
    
    def sample_timesteps(self):
        return torch.randint(1 , self.timesteps)
    
def load_plot_image_and_noise(img_path):              
 
    img = Image.open(img_path)     
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])     
    img = transform(img)  
    
    diff = Diffusion(0.0001, 0.02, 1000)         
    noise, noisy_img = diff(img, 900)  
    
    original_img = img.permute(1, 2, 0).numpy()  
    noisy_img = noisy_img.squeeze().permute(1, 2, 0).numpy()  
    
    original_img = np.clip(original_img, 0, 1)
    noisy_img = np.clip(noisy_img, 0, 1)
       
    fig , (ax1 , ax2) = plt.subplots(1 , 2 , figsize =  (10,5))
    
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
            
    ax2.imshow(noisy_img)
    ax2.set_title('Noisy Image')
    
    plt.tight_layout()     
    plt.show()          
    
    return {         
        'original_tensor': img,         
        'noise': noise,         
        'noisy_image': noisy_img     
    }       

k = load_plot_image_and_noise('E:/Diffusion-Model-From-Scratch/image (1).jpg')
print(k)


class Upsample(nn.Modlue):
    def __init__()


class UNET(nn.Module):
    def __init__(self , conv_block):
        super().__init__()
        
        self.conv1 = conv_block()
        self.conv2 = conv_block()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        
    def forward(self , x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1
        x = self.fc2