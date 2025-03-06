import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ctrgcn import Model
# NTU RGB+D 120的骨架连接对
BONES = (
(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
)

# 动作标签（NTU RGB+D 120）
ACTION_LABELS = {
    0: "drink water",  # A1
    1: "eat meal",  # A2
    2: "brush teeth",  # A3
    3: "brush hair",  # A4
    4: "drop",  # A5
    5: "pick up",  # A6
    6: "throw",  # A7
    7: "sit down",  # A8
    8: "stand up",  # A9
    9: "clapping",  # A10
    10: "reading",  # A11
    11: "writing",  # A12
    12: "tear up paper",  # A13
    13: "put on jacket",  # A14
    14: "take off jacket",  # A15
    15: "put on a shoe",  # A16
    16: "take off a shoe",  # A17
    17: "put on glasses",  # A18
    18: "take off glasses",  # A19
    19: "put on a hat/cap",  # A20
    20: "take off a hat/cap",  # A21
    21: "cheer up",  # A22
    22: "hand waving",  # A23
    23: "kicking something",  # A24
    24: "reach into pocket",  # A25
    25: "hopping",  # A26
    26: "jump up",  # A27
    27: "phone call",  # A28
    28: "play with phone/tablet",  # A29
    29: "type on a keyboard",  # A30
    30: "point to something",  # A31
    31: "taking a selfie",  # A32
    32: "check time (from watch)",  # A33
    33: "rub two hands",  # A34
    34: "nod head/bow",  # A35
    35: "shake head",  # A36
    36: "wipe face",  # A37
    37: "salute",  # A38
    38: "put palms together",  # A39
    39: "cross hands in front",  # A40
    40: "sneeze/cough",  # A41
    41: "staggering",  # A42
    42: "falling down",  # A43
    43: "headache",  # A44
    44: "chest pain",  # A45
    45: "back pain",  # A46
    46: "neck pain",  # A47
    47: "nausea/vomiting",  # A48
    48: "fan self",  # A49
    49: "punch/slap",  # A50
    50: "kicking",  # A51
    51: "pushing",  # A52
    52: "pat on back",  # A53
    53: "point finger",  # A54
    54: "hugging",  # A55
    55: "giving object",  # A56
    56: "touch pocket",  # A57
    57: "shaking hands",  # A58
    58: "walking towards",  # A59
    59: "walking apart",  # A60
    60: "put on headphone",  # A61
    61: "take off headphone",  # A62
    62: "shoot at basket",  # A63
    63: "bounce ball",  # A64
    64: "tennis bat swing",  # A65
    65: "juggle table tennis ball",  # A66
    66: "hush",  # A67
    67: "flick hair",  # A68
    68: "thumb up",  # A69
    69: "thumb down",  # A70
    70: "make OK sign",  # A71
    71: "make victory sign",  # A72
    72: "staple book",  # A73
    73: "counting money",  # A74
    74: "cutting nails",  # A75
    75: "cutting paper",  # A76
    76: "snap fingers",  # A77
    77: "open bottle",  # A78
    78: "sniff/smell",  # A79
    79: "squat down",  # A80
    80: "toss a coin",  # A81
    81: "fold paper",  # A82
    82: "ball up paper",  # A83
    83: "play magic cube",  # A84
    84: "apply cream on face",  # A85
    85: "apply cream on hand",  # A86
    86: "put on bag",  # A87
    87: "take off bag",  # A88
    88: "put object into bag",  # A89
    89: "take object out of bag",  # A90
    90: "open a box",  # A91
    91: "move heavy objects",  # A92
    92: "shake fist",  # A93
    93: "throw up cap/hat",  # A94
    94: "capitulate",  # A95
    95: "cross arms",  # A96
    96: "arm circles",  # A97
    97: "arm swings",  # A98
    98: "run on the spot",  # A99
    99: "butt kicks",  # A100
    100: "cross toe touch",  # A101
    101: "side kick",  # A102
    102: "yawn",  # A103
    103: "stretch oneself",  # A104
    104: "blow nose",  # A105
    105: "hit with object",  # A106
    106: "wield knife",  # A107
    107: "knock over",  # A108
    108: "grab stuff",  # A109
    109: "shoot with gun",  # A110
    110: "step on foot",  # A111
    111: "high-five",  # A112
    112: "cheers and drink",  # A113
    113: "carry object",  # A114
    114: "take a photo",  # A115
    115: "follow",  # A116
    116: "whisper",  # A117
    117: "exchange things",  # A118
    118: "support somebody",  # A119
    119: "rock-paper-scissors",  # A120
}

class SkeletonVisualizer:
    def __init__(self, skeleton_path, model_path, labels):
        self.data = self.load_skeleton(skeleton_path)
        self.model = self.load_model(model_path)
        self.labels = labels
        self.pred_label = self.predict_action()
        self.frame_count = self.data.shape[2]
        
    def load_skeleton(self, path):
        """解析.skeleton文件"""
        frames = []
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Skeleton file not found: {path}")
            
        frame_count = int(lines[0].strip())
        ptr = 1
        
        for frame_idx in range(frame_count):
            if ptr >= len(lines):
                print(f"Warning: Reached end of file before processing all {frame_count} frames")
                break
                
            body_count = int(lines[ptr].strip())
            ptr += 1
            
            if body_count == 0:
                ptr += 27  # Skip this frame (body info + 25 joints + joint count)
                continue
                
            # Skip body info line
            ptr += 1
            
            # Read number of joints
            joint_count = int(lines[ptr].strip())
            ptr += 1
            
            if joint_count != 25:
                print(f"Warning: Frame {frame_idx} has {joint_count} joints, expected 25. Skipping.")
                ptr += joint_count
                continue
                
            joints = []
            try:
                for joint_idx in range(25):
                    line = lines[ptr].strip()
                    if not line:
                        raise ValueError(f"Empty joint data at frame {frame_idx}, joint {joint_idx}")
                    parts = list(map(float, line.split()))
                    if len(parts) < 3:
                        raise ValueError(f"Invalid joint data at frame {frame_idx}, joint {joint_idx}: {line}")
                    joints.append(parts[:3])  # Take only x,y,z
                    ptr += 1
                
                joints_array = np.array(joints).T  # (3, 25)
                frames.append(joints_array)
            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping frame {frame_idx} due to error: {str(e)}")
                ptr += 25 - len(joints)  # Skip remaining joints
                continue
        
        if not frames:
            raise ValueError("No valid frames found in skeleton file")
            
        data = np.stack(frames, axis=2)[np.newaxis, ..., np.newaxis]  # (1, 3, T, 25, M)
        return self.normalize_data(data)
    
    def normalize_data(self, data):
        """数据归一化处理，并将 M 固定为 2"""
        N, C, T, V, M = data.shape  # 获取输入数据的 M
        print(N, C, T, V, M)

        # 如果 M=1，填充数据
        if M == 1:
            padding = data[:, :, :, :, 0:1]  # 复制第一人的数据
            data = torch.cat([torch.FloatTensor(data), torch.FloatTensor(padding)], dim=-1)  # 在 M 维度上拼接

        # 归一化处理
        center = data[:, :, :, 0:1, :]  # (N,3,T,1,2)
        data = data - center

        spine_len = np.linalg.norm(data[:, :, :, 9:10, :], axis=1)
        data = data / (spine_len + 1e-6)

        # 调整输入数据的维度顺序为 (N, C, T, V, M)
        data = data.permute(0, 1, 3, 2, 4)  # 将 V 和 T 交换

        print("Normalized data shape:", data.shape)  # 打印归一化后的数据维度
        return torch.FloatTensor(data)
    
    def load_model(self, model_path):
        """加载预训练模型"""
        try:
            model = Model(num_class=120, in_channels=3, graph="graph.ntu_rgb_d.Graph")
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def predict_action(self):
        print("predict_action data shape:", self.data.shape)  # 打印归一化后的数据维度

        """执行动作识别预测"""
        with torch.no_grad():
            output = self.model(self.data)
        pred = output.argmax(dim=1).item()
        return self.labels.get(pred, f"Unknown action ({pred})")
    
    def create_frame(self, frame, output_path):
        """创建单帧并保存为PNG"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=15, azim=-60)
        
        joints = self.data[0, :, frame, :, 0].numpy().T  # (25,3)
        
        ax.scatter(joints[:,0], joints[:,1], joints[:,2], 
                c='blue', s=50, alpha=0.6)
        
        for (i, j) in BONES:
            # 调整索引从 0 开始
            i -= 1
            j -= 1
            
            # 检查索引是否有效
            if 0 <= i < joints.shape[0] and 0 <= j < joints.shape[0]:
                ax.plot([joints[i,0], joints[j,0]],
                    [joints[i,1], joints[j,1]],
                    [joints[i,2], joints[j,2]], 
                    'r-', linewidth=2)
            else:
                print(f"Warning: Invalid bone pair ({i+1}, {j+1}) in frame {frame}")
        
        ax.text2D(0.05, 0.95, f"Action: {self.pred_label}", 
                transform=ax.transAxes, fontsize=14,
                color='red', weight='bold')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        output_file = f"{output_path}/frame_{frame:04d}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def visualize(self, output_dir="output_frames"):
        """生成所有帧并保存为PNG"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for frame in range(self.frame_count):
            self.create_frame(frame, output_dir)
        print(f"Saved {self.frame_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton', type=str, required=True,
                       help='Path to .skeleton file')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='output_frames',
                       help='Output directory for PNG frames')
    args = parser.parse_args()
    
    visualizer = SkeletonVisualizer(args.skeleton, args.model, ACTION_LABELS)
    visualizer.visualize(args.output)