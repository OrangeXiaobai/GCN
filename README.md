# README



## 环境

`Ubuntu 22.04.4 LTS`

`RTX 4090D(24GB) `

`Cuda 12.4`

`Python 3.12.9`

`PyTorch 2.6.0`

Python依赖：`requirements.txt`





## 项目结构

```
./config/ # 配置文件
|-- nturgbd-cross-subject		# 将数据（NTU RGB+D 60）按 受试者 划分训练集和测试集，测试模型对未见过的人的泛化能力
|   `-- default.yaml
|-- nturgbd-cross-view			# 将数据（NTU RGB+D 60）按 视角 划分训练集和测试集，测试模型对不同拍摄角度的适应性。
|   `-- default.yaml
|-- nturgbd120-cross-set		# 将数据（NTU RGB+D 120）按 受试者 划分训练集和测试集，测试模型对未见过的人的泛化能力
|   `-- default.yaml 
|-- nturgbd120-cross-subject	# 将数据（NTU RGB+D 120）按 视角 划分训练集和测试集，测试模型对不同拍摄角度的适应性。
|   `-- default.yaml
|-- ucla						# 可将数据（NW-UCLA）按 骨架/动作/骨架+动作 划分
	   `-- default.yaml
```

```
./data/
|-- NW-UCLA 						# 存储NW-UCLA数据集的相关文件
|   |-- all_sqe						# NW-UCLA 解压到此处
|   `-- val_label.pkl				# 存储验证集的标签（动作类别和对应的样本ID）
|-- ntu 							# 处理NTU RGB+D 60数据集的预处理脚本和统计文件。
|       `-- ......
|-- ntu120 							# 处理NTU RGB+D 120数据集的预处理脚本和统计文件。
|   |-- NTU120_CSub.npz				# 预处理后的数据文件，按受试者划分
|   |-- NTU120_CSet.npz				# 预处理后的数据文件，按视角划分
|   |-- denoised_data				# 去噪过程的中间文件和日志
|   |   `-- ......
|   |-- get_raw_denoised_data.py	# 去噪
|   |-- get_raw_skes_data.py		# 提取骨架
|   |-- raw_data					# 日志信息等
|   |   `-- ......
|   |-- seq_transformation.py		# 生成模型输入
|   `-- statistics					# 标签信息等
|       `-- ......
`-- nturgbd_raw
    |-- nturgb+d_skeletons		# NTU RGB+D 60 解压到此处
    `-- nturgb+d_skeletons120	# NTU RGB+D 120 解压到此处
```

```
./feeders/			# 从预处理后的数据集中读取数据并为模型提供输入
|-- bone_pairs.py	# 定义骨架的骨骼对（关节连接）
|-- feeder_ntu.py	# NTU数据集的数据加载器
|-- feeder_ucla.py	# ucla数据集的数据加载器
`-- tools.py		# 数据处理的辅助工具
```

```
./graph/			# 定义骨架关节之间的连接关系（邻接矩阵），为图卷积操作提供基础
|-- ntu_rgb_d.py	# NTU RGB+D数据集的图结构
|-- tools.py		# 图处理辅助工具
`-- ucla.py			# ucla数据集的图结构
```

```
./model/			# 存储模型定义的目录
|-- baseline.py		# 实现基本的时空图卷积网络
`-- ......
```

```
./torchlight/		# 定义PyTorch一些工具
	`-- ......
```

```
./work_dir/			# 记录训练信息，权重文件
	`-- ......
```



## 数据集

| **属性**          | **NTU RGB+D 60**                                             | **NTU RGB+D 120**                                            | **NW-UCLA**                                                |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| **发布年份**      | 2016                                                         | 2019                                                         | 2014                                                       |
| **动作类别数**    | 60                                                           | 120                                                          | 10                                                         |
| **样本数**        | 56,880                                                       | 114,480                                                      | ≈1,494                                                     |
| **受试者数**      | 40                                                           | 106                                                          | 10                                                         |
| **关节数**        | 25                                                           | 25                                                           | 20                                                         |
| **数据维度**      | 3D (x, y, z)                                                 | 3D (x, y, z)                                                 | 3D (x, y, z)                                               |
| **采集设备**      | Microsoft Kinect v2                                          | Microsoft Kinect v2                                          | Microsoft Kinect v1                                        |
| **帧率**          | 30 FPS                                                       | 30 FPS                                                       | 30 FPS                                                     |
| **视角数**        | 3 (不同摄像头角度)                                           | 3 (不同摄像头角度)                                           | 3 (不同摄像头角度)                                         |
| **数据类型**      | RGB视频、深度图、3D骨架、红外视频                            | RGB视频、深度图、3D骨架、红外视频                            | RGB视频、深度图、3D骨架                                    |
| **评估协议**      | 跨主体 (Cross-Subject) <br> 跨视角 (Cross-View)              | 跨主体 (Cross-Subject) <br> 跨设置 (Cross-Setup)             | 跨视角 (Cross-View)                                        |
| **训练/测试划分** | 跨主体: 40,320 / 16,560 <br> 跨视角: 37,920 / 18,960         | 跨主体: 63,026 / 51,454 <br> 跨设置: 57,307 / 57,173         | 跨视角: 1st+2nd视角 / 3rd视角 (≈1,000 / ≈494)              |
| **文件格式**      | .npz (预处理骨架数据)                                        | .npz (预处理骨架数据)                                        | .mat 或 .skeleton (预处理骨架数据)                         |
| **主要挑战**      | 多视角、双人交互、动作相似性                                 | 更多动作类别、采集设置变化                                   | 数据量少、视角变化                                         |
| **典型应用**      | 动作识别、人机交互                                           | 动作识别、行为分析                                           | 动作识别、跨视角泛化                                       |
| **下载链接**      | https://drive.google.com/file/d/1CUZnBtYwifVXS21yVg62T-vrPVayso5H/view | https://drive.google.com/file/d/1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w/view | https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0 |





## 数据处理

```
 cd ./data/ntu120
 # 提取骨架
 python get_raw_skes_data.py
 # 去噪
 python get_raw_denoised_data.py
 # 生成模型输入
 python seq_transformation.py
```





## 训练

```
nohup python main.py --config config/nturgbd120-cross-subject/actgcn_1.yaml > /dev/null 2>&1 &
```

