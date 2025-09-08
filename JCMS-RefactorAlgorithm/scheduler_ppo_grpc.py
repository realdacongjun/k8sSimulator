#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPO调度器gRPC服务，用于对接k8s simulator的PPO插件

该服务实现了PPO算法的推理功能，通过gRPC协议与k8s simulator的PPO插件对接，
提供智能调度决策能力。
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from concurrent import futures
import grpc
import sys
import os

# 添加当前目录到Python路径，以便导入生成的protobuf文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入生成的protobuf文件
try:
    import scheduler_pb2
    import scheduler_pb2_grpc
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False
    print("警告: 无法导入scheduler_pb2和scheduler_pb2_grpc模块，gRPC服务将不可用")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------
# PPO网络定义
# ---------------------
class Actor(nn.Module):
    """PPO Actor网络"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 动作空间定义（与advanced_demo.py保持一致）：
        # 0 到 N-1: 选择对应索引的节点进行调度
        # N 到 2N-1: 选择对应索引的节点进行抢占调度
        # 2N: 延迟调度动作
        # 2N+1: 拒绝调度动作
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        logits = self.net(x)
        return logits

class Critic(nn.Module):
    """PPO Critic网络"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# ---------------------
# PPO服务类
# ---------------------
class PPOService:
    """PPO服务类，提供推理功能"""
    
    def __init__(self, state_dim=20, action_dim=10, checkpoint_path='./ppo_checkpoint.pth'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_path = checkpoint_path
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # 加载检查点（如果存在）
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
        else:
            logger.info("No checkpoint found, using random initialization")
    
    def get_action(self, state, deterministic=True):
        """
        获取动作
        
        Args:
            state: 状态向量
            deterministic: 是否使用确定性策略
            
        Returns:
            int: 动作编号
        """
        # 检查状态维度是否匹配
        if len(state) != self.state_dim:
            raise ValueError(f"State dimension mismatch. Expected: {self.state_dim}, Got: {len(state)}")
        
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            if deterministic:
                action = logits.argmax(dim=1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
        return action
    
    def save_checkpoint(self):
        """保存检查点"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, self.checkpoint_path)
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")
    
    def load_checkpoint(self):
        """加载检查点"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        logger.info("Checkpoint loaded successfully")

# ---------------------
# gRPC服务实现
# ---------------------
if PROTO_AVAILABLE:
    class SchedulerServicer(scheduler_pb2_grpc.SchedulerServicer):
        """调度器gRPC服务实现"""
        
        def __init__(self, ppo_service):
            self.ppo_service = ppo_service
        
        def Predict(self, request, context):
            """
            预测接口
            
            Args:
                request: StateRequest请求
                context: gRPC上下文
                
            Returns:
                ActionResponse响应
            """
            try:
                # 获取状态向量
                state = np.array(request.state, dtype=np.float32)
                logger.info(f"收到调度请求，状态向量长度: {len(state)}")
                
                # 获取动作
                action = self.ppo_service.get_action(state)
                logger.info(f"调度决策: 动作编号 {action}")
                
                # 返回调度决策
                return scheduler_pb2.ActionResponse(action=action)
                
            except Exception as e:
                logger.error(f"处理调度请求时发生错误: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return scheduler_pb2.ActionResponse()

# ---------------------
# 服务启动函数
# ---------------------
def serve(host='[::]', port=50051, state_dim=20, action_dim=10):
    """
    启动gRPC服务
    
    Args:
        host: 服务监听主机
        port: 服务监听端口
        state_dim: 状态维度
        action_dim: 动作维度 (对于n个节点的集群，动作维度应为2*n+2)
    """
    if not PROTO_AVAILABLE:
        logger.error("无法启动gRPC服务，因为缺少protobuf模块")
        return
    
    # 初始化PPO服务
    ppo_service = PPOService(state_dim=state_dim, action_dim=action_dim)
    logger.info(f"PPO服务初始化完成，状态维度: {state_dim}，动作维度: {action_dim}")
    
    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(
        SchedulerServicer(ppo_service), server)
    
    # 添加健康检查服务（如果可用）
    try:
        from grpc_health.v1 import health
        from grpc_health.v1 import health_pb2
        from grpc_health.v1 import health_pb2_grpc
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    except ImportError:
        logger.warning("无法添加健康检查服务，缺少grpc_health模块")
    
    # 启动服务
    server.add_insecure_port(f'{host}:{port}')
    server.start()
    logger.info(f"gRPC服务启动，监听 {host}:{port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("正在关闭服务...")
        server.stop(0)

# ---------------------
# 主函数
# ---------------------
if __name__ == '__main__':
    if not PROTO_AVAILABLE:
        print("错误: 无法导入必要的protobuf模块，请确保已生成scheduler_pb2.py和scheduler_pb2_grpc.py文件")
        sys.exit(1)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PPO调度器gRPC服务')
    parser.add_argument('-H', '--host', type=str, default='[::]',
                        help='服务监听主机 (默认: [::])')
    parser.add_argument('-p', '--port', type=int, default=50051,
                        help='服务监听端口 (默认: 50051)')
    parser.add_argument('--state_dim', type=int, required=True,
                        help='状态维度 (必须指定)')
    parser.add_argument('--action_dim', type=int, required=True,
                        help='动作维度 (必须指定，对于n个节点的集群，动作维度应为2*n+2)')
    
    args = parser.parse_args()
    
    # 启动服务
    serve(args.host, args.port, args.state_dim, args.action_dim)