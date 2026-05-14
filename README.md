Oringne:# MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer (IEEE TIP 2022).

This is the official implementation of the MATR model proposed in the paper ([MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer](https://ieeexplore.ieee.org/document/9844446)) with Pytorch.

<h1 dir="auto"><a id="user-content-requirements" class="anchor" aria-hidden="true" href="#requirements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Requirements</h1>
<ul dir="auto">
<li>CUDA 11.4</li>
<li>conda 4.10.1</li>
<li>Python 3.8.12</li>
<li>PyTorch 1.9.1</li>
<li>timm 0.4.12</li>
<li>tqdm</li>
<li>glob</li>
<li>pandas</li>
</ul>

# Tips:
<strong>Dealing with RGB input:</strong>
Refer to [DPCN-Fusion](https://github.com/tthinking/DPCN-Fusion/blob/master).

<strong>Dataset is </strong> [here](http://www.med.harvard.edu/AANLIB/home.html).

The code for <strong>evaluation metrics</strong> is [here](https://github.com/tthinking/MATR/tree/main/evaluation).


# Cite the paper
If this work is helpful to you, please cite it as:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@ARTICLE{Tang_2022_MATR,
  author={Tang, Wei and He, Fazhi and Liu, Yu and Duan, Yansong},
  journal={IEEE Transactions on Image Processing}, 
  title={MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer}, 
  year={2022},
  volume={31},
  number={},
  pages={5134-5149},
  doi={10.1109/TIP.2022.3193288}}"><pre class="notranslate"><code>@ARTICLE{Tang_2022_MATR,
  author={Tang, Wei and He, Fazhi and Liu, Yu and Duan, Yansong},
  journal={IEEE Transactions on Image Processing}, 
  title={MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer}, 
  year={2022},
  volume={31},
  number={},
  pages={5134-5149},
  doi={10.1109/TIP.2022.3193288}}
</code></pre></div>

If you have any questions,  feel free to contact me (<a href="mailto:weitang2021@whu.edu.cn">weitang2021@whu.edu.cn</a>).
# Thanks for this article and for the paper MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer, as well as its repository. The code in this paper is reproducible: https://github.com/tthinking/MATR.

# MATR：基于多尺度自适应Transformer的多模态医学图像融合技术报告 

## 1. 引言与研究背景 

多模态医学图像融合旨在将来自不同模态图像的互补信息合并成一幅信息丰富的复合图像，这对于肿瘤分割、细胞分类和神经科学研究等临床应用具有重要意义。例如，SPECT图像能够反映生物体的代谢信息，利于肿瘤检测，但其分辨率通常较低；而MRI图像包含丰富的高分辨率解剖信息，能够清晰识别软组织。将两者融合，可以同时保留SPECT的功能代谢信息和MRI的软组织结构信息。

尽管基于深度学习（DL）的方法在图像融合领域取得了显著进展，但现有方法仍存在局限性：(i) 主要依赖卷积操作，虽擅长捕获局部模式，但在建模长距离上下文依赖方面能力有限；(ii) 通常采用单尺度网络，忽略了跨尺度的信息，可能导致重要信息丢失；(iii) 损失函数通常设计在像素级，对噪声容忍度低。为了克服这些缺陷，本文提出了一种名为MATR的新型无监督多模态医学图像融合方法。

## 2. 方法论与技术架构 

### 2.1 总体框架 
MATR设计了一个端到端的融合模型，结合了卷积神经网络（CNN）和Transformer的优势，以充分提取局部和全局互补特征。针对SPECT图像（RGB彩色）与MRI图像（单通道灰度）的通道不匹配问题，该方法首先将SPECT图像转换到YUV颜色空间，提取其Y分量与MRI图像在通道维度上进行连接，随后输入网络进行处理。最终融合结果通过YUV到RGB的逆转换生成。

### 2.2 核心网络组件 

*   **自适应卷积**：为了有效捕获全局上下文信息，MATR引入了自适应卷积来替代普通卷积。与基于特征图修改的传统方法不同，自适应卷积能够根据全局互补上下文直接调节卷积核，从而自适应地表征特征。
*   **自适应Transformer模块（ATM）**：为了进一步建模长距离依赖关系，网络采用了自适应Transformer模块。ATM包含多头自注意力机制（MSA）和多层感知机（MLP），并通过残差连接增强特征提取能力。
*   **多尺度架构**：网络架构被设计为多尺度形式，包含三个不同深度的分支。具有更多基本模块（BM）的分支可以提取更深层的特征，这种多尺度结构有助于充分利用跨尺度的互补信息，生成信息更丰富的融合结果。

### 2.3 损失函数设计 
由于缺乏真实的融合图像作为标签，MATR采用无监督训练策略，并设计了一个包含结构损失和区域互信息（RMI）损失的目标函数。

*   **结构损失 ($L_{SSIM}$)**：利用结构相似性指数（SSIM）约束融合图像与源图像在结构层面上的相似性，确保融合结果保留足够的结构细节。
*   **区域互信息损失 ($L_{RMI}$)**：从区域而非像素的角度限制信息传递，旨在避免不理想的人工伪影，进一步确保信息保留的准确性。

总损失函数定义为：$L_{total}=L_{SSIM}+L_{RMI}$。

## 3. 实验验证与性能评估 

### 3.1 实验设置 
实验采用了Harvard数据库中的354对SPECT和MRI图像，图像尺寸为 $256 \times 256$ 像素。训练集通过重叠裁剪策略进行了数据增强，裁剪块大小为 $120 \times 120$。实验在NVIDIA GeForce RTX 3090 GPU上基于PyTorch框架实现。

### 3.2 SPECT与MRI图像融合结果 
在SPECT和MRI图像融合任务上，MATR与LLF、NSST-PAPCNN、PMGI、U2Fusion、DDcGAN、EMFusion、SwinFuse等7种代表性方法进行了对比。

*   **定性评估**：视觉对比显示，MATR能够同时充分保留SPECT图像的功能代谢信息和MRI图像的结构软组织细节，且未引入明显伪影。相比之下，其他方法存在过度平滑、颜色失真或丢失结构细节等问题。
*   **定量评估**：在归一化互信息 ($Q_{MI}$)、Tsallis熵 ($Q_{TE}$) 等九个客观评价指标上，MATR在大多数指标上取得了最高的平均分值。例如，在 $Q_{NCIE}$ 和 $Q_{P}$ 指标上，MATR在所有20个测试样本上均获得最高分。

### 3.3 消融研究 
为了验证各组件的有效性，文中进行了详细的消融研究：

1.  **网络结构**：对比了完整MATR与替换普通卷积、移除ATM、单尺度及多阶段结构的版本。结果显示，完整模型在所有指标上均表现最优，证明了自适应卷积、ATM模块和多尺度结构的必要性。

2.  **损失函数**：单独使用区域级损失 ($L_{RMI}$) 或结构级损失 ($L_{SSIM}$) 会导致颜色失真或细节丢失，而结合两者的MATR则能产生准确且自然的融合效果。

### 3.4 泛化能力验证 
为了证明MATR的泛化能力，作者将其直接应用于其他生物医学图像融合任务，且未进行微调。

*   **PET与MRI图像融合**：MATR在PET和MRI图像融合任务中表现出色，能够同时利用PET图像的功能信息和MRI图像的组织细节，优于LLF、NSST-PAPCNN等对比方法。

*   **GFP与PC图像融合**：在绿色荧光蛋白（GFP）和相位对比（PC）图像融合任务中，MATR同样展现了优异的性能，能够充分保留输入图像的重要信息，进一步验证了其良好的泛化能力。

## 4. 结论 

本文提出的MATR方法通过引入自适应卷积和自适应Transformer，有效解决了现有方法在提取全局互补信息方面的不足。其多尺度架构和无监督损失函数设计，使得该方法在SPECT-MRI图像融合任务上超越了其他先进方法，并在PET-MRI和GFP-PC等其他融合任务中展示了卓越的泛化能力。该研究为临床诊断和治疗规划提供了具有实用价值的工程解决方案。