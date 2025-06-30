# 简介

wavelet scattering Unet只需学习解码器， 利用m2nist小型数据集

初步验证表明wavelet scattering Unet网络检测效果更精细，适合小样本分割场景

实验对比了Unet分割和倒置残差模块的LinkNet网络分割

# 网络结构

wavelet scattering Unet的编码器无需学习训练，通过计算wavelet scattering特征实现
   
代码中Model网络基于MobileNet倒置残差模块和LinkNet网络思想，编码阶段4次下采样，解码阶段4次上采样，恢复到原始尺寸   
   
更多介绍 coming soon ...

# 环境和使用
   coming soon ...

# 部分分割结果对比 

        Unet分割结果         ScatteringUnet分割结果

![image](./results/res_img9.png)                      ![image](./results/res_img9_scat.png) 
                                           
![image](./results/res_img257.png)                     ![image](./results/res_img257_scat.png) 
                                           
![image](./results/res_img285.png)                    ![image](./results/res_img285_scat.png) 

# 未来工作

  
# 参数文献
 



