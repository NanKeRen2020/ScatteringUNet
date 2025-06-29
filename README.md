# 简介
利用m2nist小型数据集，验证wavelet scattering Unet网络检测效果

同时对比了Unet分割和倒置残差模块的LinkNet网络分割

# 网络结构
   Model网络基于MobileNet倒置残差模块和LinkNet网络思想，编码阶段4次下采样，解码阶段4次上采样，恢复到原始尺寸   
   coming soon ...

# 环境和使用
   coming soon ...

# 部分分割结果对比 

        Unet分割结果                                              ScatteringUnet分割结果

![image](./results/res_img9.png)                      ![image](./pytorch/res/res_img9_scat.png) 

![image](./results/res_img257.png)                    ![image](./pytorch/res/res_img257_scat.png) 

![image](./results/res_img285.png)                    ![image](./pytorch/res/res_img285_scat.png) 
 



