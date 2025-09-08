#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPO调度器服务，用于对接k8s simulator的PPO插件

该服务实现了PPO算法的推理和训练功能，可以与k8s simulator的PPO插件对接，
提供智能调度决策能力。
"""

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, Request
import argparse
import logging
import json
import os
import torch.optim as optim
import time  # 添加time模块导入
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
from werkzeug.exceptions import RequestEntityTooLarge

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
    """PPO服务类，提供推理和训练功能"""
    
    def __init__(self, state_dim=16, action_dim=16, checkpoint_path='./ppo_checkpoint.pth', batch_size=256, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size  # 批量大小用于训练
        self.buffer_size = buffer_size  # 经验回放缓冲区大小
        
        # 设备配置 - 优先使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 经验回放缓冲区
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 添加线程锁以保护共享资源
        self.lock = threading.Lock()
        
        # 加载检查点（如果存在）
        if os.path.exists(self.checkpoint_path):
            try:
                self.load_checkpoint()
                logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Using random initialization")
        else:
            logger.info("No checkpoint found, using random initialization")
    
    def get_action(self, state, deterministic=True):
        """
        获取动作
        
        Args:
            state: 状态向量
            deterministic: 是否使用确定性策略
            
        Returns:
            选择的动作
        """
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作logits
        logits = self.actor(state_tensor)
        
        # 根据是否使用确定性策略选择动作
        if deterministic:
            # 确定性策略：选择logits最大的动作
            action = torch.argmax(logits, dim=-1)
            logger.info(f"Using deterministic policy. Selected action: {action.item()} (max logit: {torch.max(logits).item():.4f})")
        else:
            # 随机策略：根据概率分布采样
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logger.info(f"Using stochastic policy. Selected action: {action.item()} (prob: {torch.exp(dist.log_prob(action)).item():.4f})")
        
        return action.item()
    
    def update_model(self, states, actions, rewards, dones):
        """
        更新模型 - 支持批量经验回放
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            dones: 完成标志列表
            
        Returns:
            更新结果字典
        """
        logger.info(f"=================== MODEL UPDATE PROCESS STARTING ===================")
        logger.info(f"Receiving {len(states)} new experiences for training")
        
        # 处理大批量数据，分批添加到缓冲区以避免内存问题
        batch_process_size = 1000  # 每次处理1000个经验，避免内存峰值
        if len(states) > batch_process_size:
            logger.info(f"Processing large batch of {len(states)} experiences in chunks of {batch_process_size}")
            for i in range(0, len(states), batch_process_size):
                end_idx = min(i + batch_process_size, len(states))
                chunk_states = states[i:end_idx]
                chunk_actions = actions[i:end_idx]
                chunk_rewards = rewards[i:end_idx]
                chunk_dones = dones[i:end_idx]
                
                # 递归调用处理小批次
                self.update_model(chunk_states, chunk_actions, chunk_rewards, chunk_dones)
            
            logger.info("Large batch processing completed")
            return {"status": "large_batch_processed", "experiences": len(states)}
        
        # 使用线程锁保护共享资源
        with self.lock:
            try:
                # 将新经验添加到缓冲区
                for i in range(len(states)):
                    experience = {
                        'state': states[i],
                        'action': actions[i],
                        'reward': rewards[i],
                        'done': dones[i]
                    }
                    self.experience_buffer.append(experience)
                
                logger.info(f"Experience buffer size: {len(self.experience_buffer)}/{self.buffer_size}")
                
                # 修改训练触发条件，每64个经验触发一次更新
                min_batch_size = 64  # 固定为64个经验触发一次更新
                
                if len(self.experience_buffer) < min_batch_size:
                    logger.info(f"Not enough experiences for training. Need at least {min_batch_size}, have {len(self.experience_buffer)}")
                    return {
                        "status": "buffering", 
                        "buffer_size": len(self.experience_buffer),
                        "required": min_batch_size
                    }
                
                # 从缓冲区采样一批经验进行训练
                # 每64个经验触发一次更新
                if len(self.experience_buffer) >= min_batch_size:
                    # 优先使用最新的经验进行训练
                    if len(self.experience_buffer) >= self.batch_size:
                        batch_experiences = list(self.experience_buffer)[-self.batch_size:]
                    else:
                        # 如果缓冲区中的经验少于批量大小，则使用所有可用经验
                        batch_experiences = list(self.experience_buffer)[-min_batch_size:]
                else:
                    logger.info(f"Not enough experiences for training. Need at least {min_batch_size}, have {len(self.experience_buffer)}")
                    return {
                        "status": "buffering", 
                        "buffer_size": len(self.experience_buffer),
                        "required": min_batch_size
                    }
                    
                logger.info(f"Sampling {len(batch_experiences)} experiences for training")
                
                # 提取批量数据
                batch_states = [exp['state'] for exp in batch_experiences]
                batch_actions = [exp['action'] for exp in batch_experiences]
                batch_rewards = [exp['reward'] for exp in batch_experiences]
                batch_dones = [exp['done'] for exp in batch_experiences]
                
                # 转换为张量
                states_tensor = torch.FloatTensor(batch_states).to(self.device)
                actions_tensor = torch.LongTensor(batch_actions).to(self.device)
                rewards_tensor = torch.FloatTensor(batch_rewards).to(self.device)
                dones_tensor = torch.BoolTensor(batch_dones).to(self.device)
                
                logger.info(f"Training data shapes - States: {states_tensor.shape}, Actions: {actions_tensor.shape}, Rewards: {rewards_tensor.shape}")
                
                # 对于大数据集，进一步减小训练批次大小以避免内存问题
                max_training_batch = 64  # 限制训练时的最大批次大小
                if states_tensor.shape[0] > max_training_batch:
                    logger.info(f"Reducing training batch size from {states_tensor.shape[0]} to {max_training_batch} to manage memory")
                    # 只使用最新的经验进行训练
                    states_tensor = states_tensor[-max_training_batch:]
                    actions_tensor = actions_tensor[-max_training_batch:]
                    rewards_tensor = rewards_tensor[-max_training_batch:]
                    dones_tensor = dones_tensor[-max_training_batch:]
                    logger.info(f"Reduced training data shapes - States: {states_tensor.shape}, Actions: {actions_tensor.shape}, Rewards: {rewards_tensor.shape}")
                
                # 计算状态价值
                with torch.no_grad():
                    values = self.critic(states_tensor).squeeze()
                
                # 计算优势函数 (简化版本，使用蒙特卡洛回报)
                returns = []
                discounted_reward = 0
                # 反向计算回报
                for reward, done in zip(reversed(rewards_tensor.cpu().tolist()), reversed(dones_tensor.cpu().tolist())):
                    if done:
                        discounted_reward = 0
                    discounted_reward = reward + 0.99 * discounted_reward  # gamma=0.99
                    returns.insert(0, discounted_reward)
                
                returns_tensor = torch.FloatTensor(returns).to(self.device)
                advantages = returns_tensor - values
                
                # 标准化优势函数
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 记录训练前的损失
                logits = self.actor(states_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                old_log_probs = dist.log_prob(actions_tensor).detach()
                
                old_values = self.critic(states_tensor).squeeze()
                initial_critic_loss = nn.MSELoss()(old_values, returns_tensor).item()
                logger.info(f"Initial critic loss: {initial_critic_loss:.4f}")
                
                # PPO更新 (执行4个epoch)
                for epoch in range(4):
                    # 计算新的动作概率
                    logits = self.actor(states_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions_tensor)
                    
                    # 计算比率
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    
                    # 计算PPO损失
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages  # epsilon=0.2
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算价值函数损失
                    critic_values = self.critic(states_tensor).squeeze()
                    critic_loss = nn.MSELoss()(critic_values, returns_tensor)
                    
                    # 记录损失
                    if epoch == 0:
                        logger.info(f"Epoch {epoch+1}/4 completed - Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
                    
                    # 更新网络
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    
                    logger.info(f"Epoch {epoch+1}/4 completed - Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
                
                # 保存检查点
                self.save_checkpoint()
                logger.info("Model updated and checkpoint saved")
                
                # 记录训练后的一些统计信息
                final_actor_loss = actor_loss.item()
                final_critic_loss = critic_loss.item()
                avg_reward = np.mean(rewards_tensor.cpu().numpy())
                logger.info(f"Training completed. Avg reward: {avg_reward:.4f}, Final actor loss: {final_actor_loss:.4f}, Final critic loss: {final_critic_loss:.4f}")
                
                # 清理GPU内存
                del states_tensor, actions_tensor, rewards_tensor, dones_tensor
                del logits, dist, old_log_probs, old_values, advantages, returns
                del surr1, surr2, ratios, actor_loss, critic_loss, critic_values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    "status": "updated", 
                    "experiences": len(states),
                    "buffer_size": len(self.experience_buffer),
                    "trained_on": len(batch_experiences),
                    "avg_reward": float(avg_reward),
                    "final_actor_loss": final_actor_loss,
                    "final_critic_loss": final_critic_loss
                }
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUDA error" in str(e) or "Memory allocation failure" in str(e):
                    logger.error(f"Memory error during model update: {e}")
                    # 清理GPU内存并尝试使用CPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        # 尝试切换到CPU
                        try:
                            self.device = torch.device("cpu")
                            self.actor = self.actor.to(self.device)
                            self.critic = self.critic.to(self.device)
                            logger.info("Switched to CPU due to memory error")
                            # 重新尝试训练
                            return self.update_model(states, actions, rewards, dones)
                        except Exception as cpu_e:
                            logger.error(f"Failed to switch to CPU: {cpu_e}")
                            return {"status": "error", "message": f"Memory error: {e}"}
                    else:
                        return {"status": "error", "message": f"Memory error: {e}"}
                raise e
            except Exception as e:
                logger.error(f"Error during model update: {e}", exc_info=True)
                raise e
    
    def save_checkpoint(self):
        """保存检查点"""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
            }, self.checkpoint_path)
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self):
        """加载检查点"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise e

# ---------------------
# Flask应用初始化
# ---------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 设置最大请求体大小为200MB

# 全局PPO服务实例
ppo_service = None

# 创建线程池用于处理并发请求
executor = ThreadPoolExecutor(max_workers=4)  # 减少线程数以降低资源消耗

# ---------------------
# API路由实现
# ---------------------
@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    
    Returns:
        JSON格式的健康状态信息
    """
    # 降低健康检查日志级别，避免日志过多
    logger.debug("Health check request received")
    return jsonify({
        "status": "healthy",
        "service": "PPO Scheduler Service"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    调度预测接口
    
    请求格式:
    {
        "state": [0.1, 0.2, 0.3, ...]  // 状态向量
    }
    
    返回格式:
    {
        "action": 1  // 动作编号
    }
    
    Returns:
        JSON格式的调度决策
    """
    try:
        logger.debug("Predict request received")
        # 解析请求数据
        data = request.get_json()
        if not data or 'state' not in data:
            logger.error("Missing state in request")
            return jsonify({"error": "Missing state in request"}), 400
        
        # 获取状态向量
        state = data['state']
        logger.debug(f"Received state vector with {len(state)} elements")
        
        # 获取动作
        action = ppo_service.get_action(state)
        logger.debug(f"Selected action: {action}")
        
        # 返回调度决策
        return jsonify({"action": action})
        
    except ValueError as e:
        logger.error(f"Value error in predict request: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing predict request: {e}", exc_info=True)
        # 发生错误时返回默认动作
        return jsonify({"action": 0, "error": str(e)}), 500

@app.route('/update', methods=['POST'])
def update_model():
    """
    模型更新接口
    
    请求格式:
    {
        "states": [[...], [...], ...],     # 状态列表
        "actions": [1, 2, 3, ...],        # 动作列表
        "rewards": [0.1, 0.2, 0.3, ...],  # 奖励列表
        "dones": [false, false, true, ...] # 完成标志列表
    }
    
    Returns:
        JSON格式的更新结果
    """
    global ppo_service, executor
    
    try:
        # 直接获取请求数据
        data = request.get_json()
        logger.info("=" * 50)
        logger.info("RECEIVED MODEL UPDATE REQUEST")
        logger.info("=" * 50)
        
        # 异步处理模型更新以提高并发性能
        future = executor.submit(process_update_request, data)
        result, status_code = future.result(timeout=30)  # 设置超时时间
        
        # 在主线程中使用jsonify
        return jsonify(result), status_code
        
    except RequestEntityTooLarge as e:
        logger.error(f"Request entity too large: {e}", exc_info=True)
        logger.info("=" * 50)
        logger.info("UPDATE REQUEST FAILED - REQUEST TOO LARGE")
        logger.info("=" * 50)
        return jsonify({"error": "Request entity too large. Please reduce the amount of data being sent."}), 413
    except Exception as e:
        logger.error(f"Error processing update request: {e}", exc_info=True)
        logger.info("=" * 50)
        logger.info("UPDATE REQUEST FAILED")
        logger.info("=" * 50)
        return jsonify({"error": str(e)}), 500

def process_update_request(data):
    """处理模型更新请求的实际逻辑"""
    global ppo_service
    
    # 验证必需字段
    required_fields = ['states', 'actions', 'rewards', 'dones']
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return {"error": f"Missing required field: {field}"}, 400
    
    # 提取数据
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    
    # 验证数据格式
    if not isinstance(states, list) or not isinstance(actions, list) or not isinstance(rewards, list) or not isinstance(dones, list):
        logger.error("All fields must be lists")
        return {"error": "All fields must be lists"}, 400
        
    # 验证长度一致性
    if not (len(states) == len(actions) == len(rewards) == len(dones)):
        logger.error("All lists must have the same length")
        return {"error": "All lists must have the same length"}, 400
        
    # 验证states格式
    if not isinstance(states[0], list):
        logger.error("States should be a list of lists")
        return {"error": "States should be a list of lists"}, 400
        
    # 验证actions格式
    if not all(isinstance(a, (int, float)) for a in actions):
        logger.error("Actions should be a list of integers or floats")
        return {"error": "Actions should be a list of integers or floats"}, 400
        
    # 验证rewards格式
    if not all(isinstance(r, (int, float)) for r in rewards):
        logger.error("Rewards should be a list of numbers")
        return {"error": "Rewards should be a list of numbers"}, 400
        
    # 验证dones格式
    if not all(isinstance(d, bool) for d in dones):
        logger.error("Dones should be a list of booleans")
        return {"error": "Dones should be a list of booleans"}, 400
    
    logger.info(f"Processing update with {len(states)} state entries")
    logger.info(f"States shape: {[len(s) for s in states[:3]]}...")
    logger.info(f"Actions range: [{min(actions)}, {max(actions)}]")
    logger.info(f"Rewards range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    if len(states) > 0:
        logger.debug(f"First state shape: {len(states[0])}")
        logger.debug(f"First action: {actions[0] if len(actions) > 0 else 'N/A'}")
        logger.debug(f"First reward: {rewards[0] if len(rewards) > 0 else 'N/A'}")
        logger.debug(f"First done: {dones[0] if len(dones) > 0 else 'N/A'}")
    
    # 更新模型（现在支持批量经验回放）
    logger.info("Starting model update process...")
    result = ppo_service.update_model(states, actions, rewards, dones)
    logger.info(f"Model update completed: {result}")
    
    logger.info("=" * 50)
    logger.info("UPDATE REQUEST PROCESSED SUCCESSFULLY - Model Training Completed")
    logger.info("=" * 50)
    
    return result, 200

@app.errorhandler(413)
def too_large(e):
    """处理请求体过大的错误"""
    return jsonify({"error": "Request entity too large"}), 413

# ---------------------
# 服务启动函数
# ---------------------
def start_service(host='0.0.0.0', port=5000, state_dim=16, action_dim=16, batch_size=64, buffer_size=10000):
    """
    启动Flask服务
    
    Args:
        host: 服务监听主机
        port: 服务监听端口
        state_dim: 状态维度 (对于n个节点的集群，状态维度应为2*n+2)
        action_dim: 动作维度 (对于n个节点的集群，动作维度应为2*n+2)
        batch_size: 批量训练大小
        buffer_size: 经验回放缓冲区大小
    """
    global ppo_service
    
    # 初始化PPO服务，减小缓冲区大小以提高更新频率
    # 原来的buffer_size=10000，现在改为2000以提高更新频率
    adjusted_buffer_size = min(buffer_size, 2000)
    # 原来的batch_size=64，现在可以保持不变或适当减小
    adjusted_batch_size = min(batch_size, 128)
    
    ppo_service = PPOService(state_dim=state_dim, action_dim=action_dim, 
                            batch_size=adjusted_batch_size, buffer_size=adjusted_buffer_size)
    
    # 设置Flask应用的最大内容长度，以处理大型请求
    # 从100MB增加到200MB以处理更大的经验数据
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
    
    logger.info(f"Starting PPO service with state_dim={state_dim}, action_dim={action_dim}")
    logger.info(f"Adjusted batch_size={adjusted_batch_size}, buffer_size={adjusted_buffer_size}")
    logger.info(f"MAX_CONTENT_LENGTH set to 200MB")
    
    # 启动Flask服务
    app.run(host=host, port=port, threaded=True, debug=False)

# ---------------------
# 主函数
# ---------------------
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PPO调度器服务')
    parser.add_argument('-H', '--host', type=str, default='0.0.0.0',
                        help='服务监听主机 (默认: 0.0.0.0)')
    parser.add_argument('-p', '--port', type=int, default=5000,
                        help='服务监听端口 (默认: 5000)')
    parser.add_argument('--state_dim', type=int, required=True,
                        help='状态维度 (必须指定)')
    parser.add_argument('--action_dim', type=int, required=True,
                        help='动作维度 (必须指定，对于n个节点的集群，动作维度应为2*n+2)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批量训练大小 (默认: 64)')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='经验回放缓冲区大小 (默认: 10000)')
    
    args = parser.parse_args()
    
    # 启动服务
    start_service(args.host, args.port, args.state_dim, args.action_dim, args.batch_size, args.buffer_size)