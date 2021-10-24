模型代码基本介绍：
1. 检测阶段：
       使用mmdetection开源代码，基于SCNet检测网络。我们改进检测器的特征学习模式。加入视觉特征自适应加权策略。具体步骤请查阅global_context_head.py.

1.1
   scnet_roi_head.py文件为适应目标检测任务的改版（原为图像分割任务）。

2. 分类阶段：
       将检测信息通过语义特征自适应加权，加强建筑对象之间的联系。具体步骤请查阅land use classification。

3. buildname.txt是词向量训练文本。

	
