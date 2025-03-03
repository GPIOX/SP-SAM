import torch
import numpy as np

def extract_ampl_phase(fft_im):
    """
    从给定的傅里叶变换结果中提取幅度和相位。
    
    参数:
    fft_im: 傅里叶变换结果，大小应为 bx3xhxwx2。
    
    返回:
    fft_amp: 傅里叶变换的幅度。
    fft_pha: 傅里叶变换的相位。
    """
    # 计算幅度
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    # 计算相位
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    """
    将源幅度的低频部分替换为目标幅度的低频部分。
    
    参数:
    amp_src: 源幅度。
    amp_trg: 目标幅度。
    L: 低频范围的比例。
    
    返回:
    amp_src: 替换源幅度低频部分后的结果。
    """
    _, _, h, w = amp_src.size()

    # 计算低频范围的大小
    b = 1

    # 替换源幅度四个角的低频部分
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # 左上角
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # 右上角
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # 左下角
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # 右下角
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    """
    将源幅度的低频部分替换为目标幅度的低频部分（NumPy版本）。
    
    参数:
    amp_src: 源幅度。
    amp_trg: 目标幅度。
    L: 低频范围的比例。
    
    返回:
    a_src: 替换源幅度低频部分后的结果。
    """
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    # 计算低频范围的大小
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    # 计算幅度矩阵的中心坐标
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    # 计算低频部分的范围
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # 替换源幅度的低频部分
    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    # 反向移位
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    """
    执行频率域适应（FDA）将源图像转换为目标图像。
    
    参数:
    src_img: 源图像。
    trg_img: 目标图像。
    L: 低频范围的比例。
    
    返回:
    src_in_trg: 重组后的源图像，带有目标图像的低频风格。
    """
    # 获取源图像和目标图像的傅里叶变换
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # 提取幅度和相位
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # 替换源幅度的低频部分
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # 重组源图像的傅里叶变换
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # 获取重组后的图像：源内容，目标风格
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg