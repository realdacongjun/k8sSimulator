/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ppo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"google.golang.org/grpc"
	"k8s.io/klog"
	k8sframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"

	"sync" // 添加sync包用于并发控制
)

const (
	// PluginName indicates name of volcano scheduler plugin.
	PluginName = "ppo"

	// Default configurations
	defaultPPOServiceAddress = "localhost:5000"  // HTTP服务地址
	defaultPPOGrpcAddress    = "localhost:50051" // gRPC服务地址
	defaultTimeout           = 5 * time.Second   // 请求超时时间
)

// Experience表示一次调度经验
type Experience struct {
	State     []float64 `json:"state"`
	Action    int32     `json:"action"`
	Reward    float64   `json:"reward"`
	NextState []float64 `json:"next_state"`
	Done      bool      `json:"done"`
}

// ppoPlugin implements the PPO scheduling plugin
type ppoPlugin struct {
	// Arguments given for the plugin
	pluginArguments framework.Arguments

	// PPO service configurations
	httpAddress string
	grpcAddress string
	timeout     time.Duration

	// gRPC连接
	grpcConn *grpc.ClientConn
	// grpcClient scheduler_pb.SchedulerClient  // 暂时注释掉，因为缺少生成的protobuf代码

	// 经验收集
	sessionExperiences []Experience // 当前会话的经验

	// 添加互斥锁保护actionCache
	actionCache map[string]int32
	cacheMutex  sync.RWMutex
}

// New function returns ppo plugin object.
func New(arguments framework.Arguments) framework.Plugin {
	plugin := &ppoPlugin{
		pluginArguments: arguments,
		httpAddress:     defaultPPOServiceAddress,
		grpcAddress:     defaultPPOGrpcAddress,
		timeout:         defaultTimeout,
	}

	// 从参数中读取配置
	if httpAddr, ok := arguments["http-address"]; ok {
		if addr, ok := httpAddr.(string); ok {
			plugin.httpAddress = addr
		}
	}

	if grpcAddr, ok := arguments["grpc-address"]; ok {
		if addr, ok := grpcAddr.(string); ok {
			plugin.grpcAddress = addr
		}
	}

	// 初始化gRPC客户端
	plugin.initGrpcClient()

	return plugin
}

// initGrpcClient 初始化gRPC客户端连接
func (pp *ppoPlugin) initGrpcClient() {
	// 暂时注释掉gRPC相关代码，因为缺少生成的protobuf代码
	/*
		conn, err := grpc.Dial(pp.grpcAddress, grpc.WithInsecure())
		if err != nil {
			klog.Errorf("Failed to connect to PPO gRPC service at %s: %v", pp.grpcAddress, err)
			return
		}
		pp.grpcConn = conn
		// pp.grpcClient = scheduler_pb.NewSchedulerClient(conn)  // 暂时注释掉
		klog.Infof("Connected to PPO gRPC service at %s", pp.grpcAddress)
	*/
}

// Name returns the name of the plugin
func (pp *ppoPlugin) Name() string {
	return PluginName
}

// OnSessionOpen is called when the session opens
func (pp *ppoPlugin) OnSessionOpen(ssn *framework.Session) {
	klog.V(3).Infof("Enter ppo plugin, session has %d nodes and %d jobs...", len(ssn.NodeList), len(ssn.Jobs))
	klog.V(3).Infof("PPO service HTTP address: %s, gRPC address: %s", pp.httpAddress, pp.grpcAddress)

	// 测试连接到PPO服务
	if err := pp.testPPOServiceConnection(); err != nil {
		klog.Errorf("Failed to connect to PPO service: %v", err)
	}

	// 初始化经验收集
	pp.sessionExperiences = []Experience{}

	// 用于缓存任务-节点对的PPO动作结果，避免重复计算
	pp.actionCache = make(map[string]int32)

	// 注册节点排序函数，用于获取PPO调度决策
	nodeOrderFn := func(task *api.TaskInfo, node *api.NodeInfo) (float64, error) {
		klog.V(4).Infof("PPO plugin processing task %s on node %s", task.Name, node.Name)

		// 构建状态向量
		state := pp.buildStateVector(ssn, task, node)
		klog.V(5).Infof("Built state vector with %d elements for task %s", len(state), task.Name)

		// 获取PPO动作
		action, err := pp.getActionFromPPO(state)
		if err != nil {
			klog.Errorf("Failed to get action from PPO service: %v", err)
			// 出错时返回默认分数
			return 0, nil
		}

		// 缓存动作结果，供其他回调函数使用（使用互斥锁保护）
		pp.cacheMutex.Lock()
		pp.actionCache[fmt.Sprintf("%s-%s", task.UID, node.Name)] = action
		pp.cacheMutex.Unlock()

		klog.V(4).Infof("Got action %d from PPO service for task %s", action, task.Name)

		// 收集经验
		exp := Experience{
			State:  state,
			Action: action,
			// Reward会在会话结束时计算
			// NextState将在下一个调度步骤中确定
			Done: false,
		}

		// 如果已有经验，更新前一个经验的NextState
		pp.cacheMutex.Lock()
		if len(pp.sessionExperiences) > 0 {
			lastIndex := len(pp.sessionExperiences) - 1
			pp.sessionExperiences[lastIndex].NextState = state
		}

		pp.sessionExperiences = append(pp.sessionExperiences, exp)
		pp.cacheMutex.Unlock()
		klog.V(4).Infof("Collected experience #%d for task %s, total experiences: %d", len(pp.sessionExperiences), task.Name, len(pp.sessionExperiences))

		// 根据动作计算节点分数
		score := pp.calculateNodeScore(action, node, ssn)
		klog.V(4).Infof("Calculated score %f for task %s on node %s with action %d", score, task.Name, node.Name, action)
		return score, nil
	}

	// 添加节点排序函数
	ssn.AddNodeOrderFn(pp.Name(), nodeOrderFn)

	// 添加BatchNodeOrderFn用于批量处理节点评分
	batchNodeOrderFn := func(task *api.TaskInfo, nodes []*api.NodeInfo) (map[string]float64, error) {
		nodeScoreMap := make(map[string]float64)

		// 为每个节点计算分数，优先使用缓存的动作结果
		for _, node := range nodes {
			cacheKey := fmt.Sprintf("%s-%s", task.UID, node.Name)
			var action int32
			var foundInCache bool

			// 检查是否有缓存的动作结果（使用读锁保护）
			pp.cacheMutex.RLock()
			if cachedAction, exists := pp.actionCache[cacheKey]; exists {
				action = cachedAction
				foundInCache = true
				klog.V(5).Infof("Using cached action %d for task %s on node %s", action, task.Name, node.Name)
			}
			pp.cacheMutex.RUnlock()

			if !foundInCache {
				// 没有缓存则构建状态向量并调用PPO服务
				state := pp.buildStateVector(ssn, task, node)
				var err error
				action, err = pp.getActionFromPPO(state)
				if err != nil {
					klog.Errorf("Failed to get action from PPO service: %v", err)
					// 出错时使用默认分数
					nodeScoreMap[node.Name] = 0.0
					continue
				}
				foundInCache = false
				klog.V(5).Infof("Got new action %d for task %s on node %s", action, task.Name, node.Name)
			}

			// 根据动作计算节点分数
			score := pp.calculateNodeScore(action, node, ssn)
			nodeScoreMap[node.Name] = score

			// 如果不是从缓存获取的动作，则缓存它（使用互斥锁保护）
			if !foundInCache {
				pp.cacheMutex.Lock()
				pp.actionCache[cacheKey] = action
				pp.cacheMutex.Unlock()
			}

			klog.V(5).Infof("BatchNodeOrderFn: Task %s on node %s got action %d with score %f",
				task.Name, node.Name, action, score)
		}

		return nodeScoreMap, nil
	}

	// 添加NodeMapFn用于节点映射阶段评分
	nodeMapFn := func(task *api.TaskInfo, node *api.NodeInfo) (float64, error) {
		// 优先使用缓存的动作结果（使用读锁保护）
		cacheKey := fmt.Sprintf("%s-%s", task.UID, node.Name)
		var action int32
		var foundInCache bool

		pp.cacheMutex.RLock()
		if cachedAction, exists := pp.actionCache[cacheKey]; exists {
			action = cachedAction
			foundInCache = true
			klog.V(5).Infof("Using cached action %d for task %s on node %s", action, task.Name, node.Name)
		}
		pp.cacheMutex.RUnlock()

		if !foundInCache {
			// 没有缓存则构建状态向量并调用PPO服务
			state := pp.buildStateVector(ssn, task, node)
			var err error
			action, err = pp.getActionFromPPO(state)
			if err != nil {
				klog.Errorf("Failed to get action from PPO service: %v", err)
				// 出错时返回默认分数
				return 0.0, nil
			}
			foundInCache = false
			klog.V(5).Infof("Got new action %d for task %s on node %s", action, task.Name, node.Name)
		}

		// 根据动作计算节点分数
		score := pp.calculateNodeScore(action, node, ssn)

		// 如果不是从缓存获取的动作，则缓存它（使用互斥锁保护）
		if !foundInCache {
			pp.cacheMutex.Lock()
			pp.actionCache[cacheKey] = action
			pp.cacheMutex.Unlock()
		}

		klog.V(5).Infof("NodeMapFn: Task %s on node %s got action %d with score %f",
			task.Name, node.Name, action, score)

		return score, nil
	}

	// 添加NodeReduceFn用于节点归约阶段处理
	nodeReduceFn := func(task *api.TaskInfo, pluginNodeScoreMap k8sframework.NodeScoreList) error {
		// 在这个阶段，我们不需要额外的处理，因为每个节点的分数已经在map阶段计算完成
		// 这里只是记录日志表明归约阶段已经执行
		klog.V(5).Infof("NodeReduceFn: Processing scores for task %s, received %d node scores",
			task.Name, len(pluginNodeScoreMap))

		// 记录每个节点的分数
		for _, nodeScore := range pluginNodeScoreMap {
			klog.V(5).Infof("NodeReduceFn: Node %s score: %d", nodeScore.Name, nodeScore.Score)
		}

		return nil
	}

	ssn.AddBatchNodeOrderFn(pp.Name(), batchNodeOrderFn)
	ssn.AddNodeMapFn(pp.Name(), nodeMapFn)
	ssn.AddNodeReduceFn(pp.Name(), nodeReduceFn)

	klog.V(3).Infof("PPO plugin registered all node order functions")
}

// testPPOServiceConnection 测试与PPO服务的连接
func (pp *ppoPlugin) testPPOServiceConnection() error {
	// 测试HTTP连接
	url := fmt.Sprintf("http://%s/health", pp.httpAddress)

	client := &http.Client{
		Timeout: 3 * time.Second,
	}

	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("failed to connect to PPO HTTP service at %s: %v", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("PPO HTTP service at %s returned status %d", url, resp.StatusCode)
	}

	klog.V(3).Infof("Successfully connected to PPO HTTP service at %s", url)
	return nil
}

// buildStateVector 构建状态向量
func (pp *ppoPlugin) buildStateVector(ssn *framework.Session, task *api.TaskInfo, targetNode *api.NodeInfo) []float64 {
	// 状态向量包含以下信息:
	// 1. 每个节点的CPU和内存使用率
	// 2. 待调度任务的资源需求
	// 3. 队列信息等

	var state []float64

	// 添加每个节点的资源使用率 (使用实际节点数量，每个节点2个资源指标)
	nodeCount := len(ssn.NodeList)
	currentNodeIndex := 0

	// 遍历当前节点列表
	for _, node := range ssn.NodeList {
		if currentNodeIndex >= nodeCount {
			break // 最多处理实际节点数
		}

		// CPU使用率 = 已用CPU / 总CPU
		cpuUsage := 0.0
		if node.Allocatable.MilliCPU > 0 {
			cpuUsage = float64(node.Used.MilliCPU) / float64(node.Allocatable.MilliCPU)
		}

		// 内存使用率 = 已用内存 / 总内存
		memUsage := 0.0
		if node.Allocatable.Memory > 0 {
			memUsage = float64(node.Used.Memory) / float64(node.Allocatable.Memory)
		}

		state = append(state, cpuUsage, memUsage)
		currentNodeIndex++
	}

	// 添加任务资源需求 (标准化到0-1范围)
	// 假设最大CPU为100核(100000m), 最大内存为1TB(1073741824KB)
	taskCPUReq := 0.0
	taskMemReq := 0.0
	if task.Resreq != nil {
		taskCPUReq = float64(task.Resreq.MilliCPU) / 100000.0
		taskMemReq = float64(task.Resreq.Memory) / 1073741824.0
	}
	state = append(state, taskCPUReq, taskMemReq)

	klog.V(4).Infof("Built state vector with %d elements (nodeCount: %d)", len(state), nodeCount)
	return state
}

// getActionFromPPO 从PPO服务获取动作
func (pp *ppoPlugin) getActionFromPPO(state []float64) (int32, error) {
	// 优先尝试使用gRPC
	/*
		if pp.grpcClient != nil {
			action, err := pp.getActionFromGrpc(state)
			if err == nil {
				return action, nil
			}
			klog.Warningf("Failed to get action from gRPC, fallback to HTTP: %v", err)
		}
	*/

	// 回退到HTTP
	return pp.getActionFromHTTP(state)
}

// getActionFromGrpc 通过gRPC从PPO服务获取动作
/*
func (pp *ppoPlugin) getActionFromGrpc(state []float64) (int32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), pp.timeout)
	defer cancel()

	// 构造请求
	request := &scheduler_pb.StateRequest{
		State: state,
	}

	// 发送请求
	response, err := pp.grpcClient.Predict(ctx, request)
	if err != nil {
		return 0, fmt.Errorf("gRPC call failed: %v", err)
	}

	return response.Action, nil
}
*/

// getActionFromHTTP 通过HTTP从PPO服务获取动作
func (pp *ppoPlugin) getActionFromHTTP(state []float64) (int32, error) {
	klog.V(5).Infof("Getting action from PPO service via HTTP, state length: %d", len(state))

	// 构造请求数据
	requestData := map[string]interface{}{
		"state": state,
	}

	// 序列化为JSON
	jsonData, err := json.Marshal(requestData)
	if err != nil {
		klog.Errorf("Failed to marshal request data: %v", err)
		return 0, fmt.Errorf("failed to marshal request data: %v", err)
	}

	klog.V(5).Infof("Sending state to PPO service: %v", state)

	// 创建HTTP请求
	url := fmt.Sprintf("http://%s/predict", pp.httpAddress)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		klog.Errorf("Failed to create HTTP request: %v", err)
		return 0, fmt.Errorf("failed to create HTTP request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// 设置超时
	client := &http.Client{
		Timeout: pp.timeout,
	}

	// 发送请求
	resp, err := client.Do(req)
	if err != nil {
		klog.Errorf("HTTP request failed: %v", err)
		return 0, fmt.Errorf("HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	klog.V(5).Infof("Received response from PPO service, status code: %d", resp.StatusCode)

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		klog.Errorf("HTTP request failed with status: %d", resp.StatusCode)
		return 0, fmt.Errorf("HTTP request failed with status: %d", resp.StatusCode)
	}

	// 解析响应
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		klog.Errorf("Failed to decode response: %v", err)
		return 0, fmt.Errorf("failed to decode response: %v", err)
	}

	// 获取动作
	action, ok := result["action"].(float64)
	if !ok {
		klog.Errorf("Invalid response format: action not found or invalid type, response: %v", result)
		return 0, fmt.Errorf("invalid response format: action not found or invalid type")
	}

	klog.V(4).Infof("Successfully got action %d from PPO service", int32(action))
	return int32(action), nil
}

// calculateNodeScore 根据PPO动作计算节点分数
func (pp *ppoPlugin) calculateNodeScore(action int32, node *api.NodeInfo, ssn *framework.Session) float64 {
	nodeCount := len(ssn.NodeList) // 使用实际的节点数量而不是硬编码

	// 动作空间定义：
	// 0 到 N-1: 选择对应索引的节点进行调度
	// N 到 2N-1: 选择对应索引的节点进行抢占调度
	// 2N: 延迟调度动作
	// 2N+1: 拒绝调度动作

	// 检查节点列表长度
	if nodeCount == 0 {
		klog.Warning("NodeList is empty. Using default score.")
		return 0
	}

	// 检查action是否在有效范围内
	if int(action) < 0 || int(action) >= 2*nodeCount+2 {
		klog.Warningf("Invalid action received: %d, nodeCount: %d. Using default score.", action, nodeCount)
		return 0
	}

	if int(action) < nodeCount {
		// 直接调度动作
		// 确保索引不越界
		if int(action) >= len(ssn.NodeList) {
			klog.Warningf("Action index %d is out of NodeList range %d. Using default score.", action, len(ssn.NodeList))
			return 0
		}

		if node.Name == ssn.NodeList[action].Name {
			return 100 // 高分表示首选节点
		}
		return 0 // 低分表示不选择该节点
	} else if int(action) < 2*nodeCount {
		// 抢占调度动作
		preemptIndex := int(action) - nodeCount
		// 确保索引不越界
		if preemptIndex >= len(ssn.NodeList) {
			klog.Warningf("Preempt index %d is out of NodeList range %d. Using default score.", preemptIndex, len(ssn.NodeList))
			return 0
		}

		if node.Name == ssn.NodeList[preemptIndex].Name {
			return 50 // 中等分数表示抢占调度
		}
		return 0
	} else if int(action) == 2*nodeCount {
		// 延迟调度动作
		return -10 // 负分表示延迟调度
	} else {
		// 拒绝调度动作
		return -100 // 最低分表示拒绝调度
	}
}

// OnSessionClose is called when the session closes
func (pp *ppoPlugin) OnSessionClose(ssn *framework.Session) {
	klog.V(3).Infof("PPO plugin session closing with %d experiences collected", len(pp.sessionExperiences))

	// 标记最后一个经验为完成（使用互斥锁保护）
	pp.cacheMutex.Lock()
	if len(pp.sessionExperiences) > 0 {
		pp.sessionExperiences[len(pp.sessionExperiences)-1].Done = true
		klog.V(4).Infof("Marked last experience as done")
	}
	pp.cacheMutex.Unlock()

	// 计算奖励
	klog.V(4).Infof("Calculating rewards...")
	pp.calculateRewards(ssn)

	// 发送经验到PPO服务进行训练
	pp.cacheMutex.RLock()
	experiencesCount := len(pp.sessionExperiences)
	pp.cacheMutex.RUnlock()

	if experiencesCount > 0 {
		klog.V(3).Infof("Sending %d experiences for training", experiencesCount)

		// 创建临时切片以避免长时间持有锁
		pp.cacheMutex.RLock()
		tempExperiences := make([]Experience, len(pp.sessionExperiences))
		copy(tempExperiences, pp.sessionExperiences)
		pp.cacheMutex.RUnlock()

		err := pp.sendExperiencesForTraining(tempExperiences)
		if err != nil {
			klog.Errorf("Failed to send experiences for training: %v", err)
		} else {
			klog.V(3).Infof("Successfully sent %d experiences for training", experiencesCount)
		}
	} else {
		klog.V(3).Infof("No experiences to send for training")
	}

	// 清理资源
	if pp.grpcConn != nil {
		pp.grpcConn.Close()
	}

	klog.V(4).Infof("Leaving ppo plugin ...")
}

// calculateRewards 根据会话统计信息计算每个经验的奖励
func (pp *ppoPlugin) calculateRewards(ssn *framework.Session) {
	klog.V(4).Infof("Starting reward calculation with session data:")
	klog.V(4).Infof("Total experiences: %d", len(pp.sessionExperiences))
	klog.V(4).Infof("Session has %d jobs and %d nodes", len(ssn.Jobs), len(ssn.NodeList))

	// 基于多种因素计算奖励
	totalTasks := 0
	totalAllocatedTasks := 0
	// completedTasks := 0
	totalJCT := 0.0           // 总任务完成时间
	totalWaitingTime := 0.0   // 总等待时间
	totalResourceUsage := 0.0 // 总资源使用

	// 新增变量用于资源匹配度和大任务奖励计算
	totalResourceMatchScore := 0.0 // 总资源匹配度分数
	highResourceTaskCount := 0     // 高资源需求任务数
	highResourceTaskCompleted := 0 // 高资源需求任务完成数
	failedTaskCount := 0           // 失败任务数

	// 遍历所有任务统计信息
	for _, job := range ssn.Jobs {
		totalTasks += len(job.Tasks)

		// 计算任务完成时间和等待时间
		for _, task := range job.Tasks {
			// 检查任务状态
			if task.Status == api.Running || task.Status == api.Succeeded {
				// 正在运行或成功完成的任务
				totalAllocatedTasks++

				// 检查是否为高资源需求任务
				if isHighResourceTask(task) {
					highResourceTaskCount++
					if task.Status == api.Succeeded {
						highResourceTaskCompleted++
					}
				}
			} else if task.Status == api.Failed {
				// 失败的任务
				failedTaskCount++

				// 检查是否为高资源需求任务
				if isHighResourceTask(task) {
					highResourceTaskCount++
				}
			}

			// 计算资源使用情况和匹配度
			if task.Resreq != nil {
				totalResourceUsage += float64(task.Resreq.MilliCPU) / 1000.0
				totalResourceUsage += float64(task.Resreq.Memory) / (1024 * 1024 * 1024)

				// 计算任务与分配节点的资源匹配度
				if task.NodeName != "" {
					for _, node := range ssn.NodeList {
						if node.Name == task.NodeName {
							matchScore := calculateResourceMatchScore(task, node)
							totalResourceMatchScore += matchScore

							// 打印每个任务的资源匹配信息
							klog.V(3).Infof("Task %s on node %s: CPU req=%d m, mem req=%d bytes, node CPU allocatable=%d m, node mem allocatable=%d bytes, match score=%.4f",
								task.Name, node.Name, task.Resreq.MilliCPU, task.Resreq.Memory,
								node.Allocatable.MilliCPU, node.Allocatable.Memory, matchScore)
							break
						}
					}
				}
			}
		}
	}

	klog.V(3).Infof("Task statistics: total=%d, allocated=%d, highResource=%d, highResourceCompleted=%d, failed=%d",
		totalTasks, totalAllocatedTasks, highResourceTaskCount, highResourceTaskCompleted, failedTaskCount)

	// 计算各种奖励分量
	completionReward := 0.0
	if totalTasks > 0 {
		completionRate := float64(totalAllocatedTasks) / float64(totalTasks)
		completionReward = completionRate * 10.0 // 完成率奖励
		klog.V(3).Infof("Completion reward calculation: allocated tasks=%d, total tasks=%d, completion rate=%.4f, reward=%.4f",
			totalAllocatedTasks, totalTasks, completionRate, completionReward)

		// 添加分配任务的详细信息
		klog.V(4).Infof("Allocated tasks details:")
		for _, job := range ssn.Jobs {
			for _, task := range job.Tasks {
				if task.Status == api.Running || task.Status == api.Succeeded {
					klog.V(5).Infof("Allocated task: %s (Job: %s), Node: %s", task.Name, job.Name, task.NodeName)
				}
			}
		}
	} else {
		klog.V(3).Infof("No tasks found, completion reward set to 0")
		klog.V(4).Infof("Session jobs data: %v", ssn.Jobs)
	}

	waitingTimeReward := 0.0
	if totalAllocatedTasks > 0 {
		avgWaitingTime := totalWaitingTime / float64(totalAllocatedTasks)
		// 等待时间越短奖励越高(使用负指数函数)
		waitingTimeReward = math.Exp(-avgWaitingTime/100.0) * 5.0
		klog.V(3).Infof("Waiting time reward calculation: total waiting time=%.4f, avg waiting time=%.4f, reward=%.4f",
			totalWaitingTime, avgWaitingTime, waitingTimeReward)
	} else {
		klog.V(3).Infof("No allocated tasks, waiting time reward set to 0")
	}

	jctReward := 0.0
	if totalAllocatedTasks > 0 {
		avgJCT := totalJCT / float64(totalAllocatedTasks)
		// 任务完成时间越短奖励越高
		jctReward = math.Exp(-avgJCT/100.0) * 5.0
		klog.V(3).Infof("JCT reward calculation: total JCT=%.4f, avg JCT=%.4f, reward=%.4f",
			totalJCT, avgJCT, jctReward)
	} else {
		klog.V(3).Infof("No allocated tasks, JCT reward set to 0")
	}

	resourceBalanceReward := 0.0
	// 计算资源平衡奖励(节点间资源使用越均衡奖励越高)
	if len(ssn.NodeList) > 1 {
		var nodeUsages []float64
		totalNodeUsage := 0.0

		nodeUsageDetails := ""
		for _, node := range ssn.NodeList {
			usage := 0.0
			if node.Allocatable.MilliCPU > 0 {
				usage = float64(node.Used.MilliCPU) / float64(node.Allocatable.MilliCPU)
			}
			nodeUsages = append(nodeUsages, usage)
			totalNodeUsage += usage
			nodeUsageDetails += fmt.Sprintf("%s=%.4f, ", node.Name, usage)
		}

		avgUsage := totalNodeUsage / float64(len(ssn.NodeList))
		variance := 0.0
		for _, usage := range nodeUsages {
			variance += math.Pow(usage-avgUsage, 2)
		}
		variance /= float64(len(ssn.NodeList))

		// 方差越小奖励越高
		resourceBalanceReward = math.Exp(-variance*10) * 3.0
		klog.V(3).Infof("Resource balance reward calculation: node usages [%s], avg usage=%.4f, variance=%.6f, reward=%.4f",
			nodeUsageDetails, avgUsage, variance, resourceBalanceReward)
	} else {
		klog.V(3).Infof("Less than 2 nodes, resource balance reward set to 0")
	}

	// 新增：资源匹配度奖励
	resourceMatchReward := totalResourceMatchScore * 2.0 // 资源匹配度奖励权重为2.0
	klog.V(3).Infof("Resource match reward calculation: total match score=%.4f, reward=%.4f",
		totalResourceMatchScore, resourceMatchReward)

	// 新增：大任务奖励
	highResourceTaskReward := 0.0
	if highResourceTaskCount > 0 {
		highResourceCompletionRate := float64(highResourceTaskCompleted) / float64(highResourceTaskCount)
		highResourceTaskReward = highResourceCompletionRate * 8.0 // 大任务完成奖励权重为8.0
		klog.V(3).Infof("High resource task reward calculation: completed=%d, total high resource tasks=%d, completion rate=%.4f, reward=%.4f",
			highResourceTaskCompleted, highResourceTaskCount, highResourceCompletionRate, highResourceTaskReward)
	} else {
		klog.V(3).Infof("No high resource tasks, high resource task reward set to 0")
	}

	// 新增：故障惩罚
	failurePenalty := float64(failedTaskCount) * -3.0 // 每个失败任务惩罚3分
	klog.V(3).Infof("Failure penalty calculation: failed tasks=%d, penalty=%.4f",
		failedTaskCount, failurePenalty)

	// 计算总奖励
	totalReward := completionReward + waitingTimeReward + jctReward + resourceBalanceReward +
		resourceMatchReward + highResourceTaskReward + failurePenalty

	klog.V(3).Infof("Total reward components: completion=%.4f, waiting=%.4f, jct=%.4f, balance=%.4f, match=%.4f, highResource=%.4f, failure=%.4f, TOTAL=%.4f",
		completionReward, waitingTimeReward, jctReward, resourceBalanceReward,
		resourceMatchReward, highResourceTaskReward, failurePenalty, totalReward)

	// 添加奖励计算的额外调试信息
	klog.V(4).Infof("Reward calculation details:")
	klog.V(4).Infof("Total tasks: %d, Allocated tasks: %d", totalTasks, totalAllocatedTasks)
	klog.V(4).Infof("Failed tasks: %d", failedTaskCount)
	klog.V(4).Infof("High resource tasks: %d, Completed: %d", highResourceTaskCount, highResourceTaskCompleted)
	klog.V(4).Infof("Total resource match score: %.4f", totalResourceMatchScore)

	// 为每个经验分配奖励
	experiencesWithRewards := 0
	for i := range pp.sessionExperiences {
		// 可以根据具体调度决策调整奖励
		oldReward := pp.sessionExperiences[i].Reward
		pp.sessionExperiences[i].Reward = totalReward

		// 如果是最后一个经验且任务完成，给予额外奖励
		if i == len(pp.sessionExperiences)-1 && totalAllocatedTasks == totalTasks && totalTasks > 0 {
			pp.sessionExperiences[i].Reward += 2.0
			klog.V(3).Infof("Added completion bonus of 2.0 to the last experience")
		}

		// 打印每个经验的详细奖励信息
		klog.V(3).Infof("Experience %d reward details - State length: %d, Action: %d, Old reward: %.4f, New reward: %.4f, Next state length: %d, Done: %t",
			i, len(pp.sessionExperiences[i].State), pp.sessionExperiences[i].Action,
			oldReward, pp.sessionExperiences[i].Reward, len(pp.sessionExperiences[i].NextState), pp.sessionExperiences[i].Done)

		// 只有非零奖励的经验才算作有效经验
		if pp.sessionExperiences[i].Reward != 0 {
			experiencesWithRewards++
		}
	}

	klog.V(3).Infof("Reward calculation completed. Total experiences: %d, Experiences with non-zero rewards: %d",
		len(pp.sessionExperiences), experiencesWithRewards)
}

// isHighResourceTask 判断是否为高资源需求任务
func isHighResourceTask(task *api.TaskInfo) bool {
	// 定义高资源需求任务的标准
	// 例如：CPU需求超过4核 或 内存需求超过8GB
	const highCPUThreshold = 4000                      // 4核 = 4000m
	const highMemoryThreshold = 8 * 1024 * 1024 * 1024 // 8GB

	if task.Resreq == nil {
		return false
	}

	isHighCPU := task.Resreq.MilliCPU > highCPUThreshold
	isHighMemory := task.Resreq.Memory > highMemoryThreshold

	return isHighCPU || isHighMemory
}

// calculateResourceMatchScore 计算任务与节点的资源匹配度分数
func calculateResourceMatchScore(task *api.TaskInfo, node *api.NodeInfo) float64 {
	if task.Resreq == nil || node.Allocatable == nil {
		return 0.0
	}

	// 计算CPU匹配度 (0-1之间，1表示完美匹配)
	cpuMatch := 0.0
	if node.Allocatable.MilliCPU > 0 {
		// 任务CPU需求占节点总CPU的比例
		cpuRatio := float64(task.Resreq.MilliCPU) / float64(node.Allocatable.MilliCPU)
		// 使用高斯函数计算匹配度，最优匹配在0.8左右
		cpuMatch = math.Exp(-math.Pow(cpuRatio-0.8, 2) / (2 * 0.2 * 0.2))
	}

	// 计算内存匹配度 (0-1之间，1表示完美匹配)
	memoryMatch := 0.0
	if node.Allocatable.Memory > 0 {
		// 任务内存需求占节点总内存的比例
		memoryRatio := float64(task.Resreq.Memory) / float64(node.Allocatable.Memory)
		// 使用高斯函数计算匹配度，最优匹配在0.8左右
		memoryMatch = math.Exp(-math.Pow(memoryRatio-0.8, 2) / (2 * 0.2 * 0.2))
	}

	// 综合匹配度分数 (CPU和内存匹配度的加权平均)
	matchScore := (cpuMatch*0.5 + memoryMatch*0.5) * 10 // 乘以10放大分数范围

	return matchScore
}

// sendExperiencesForTraining 将经验数据发送到PPO服务进行训练
func (pp *ppoPlugin) sendExperiencesForTraining(experiences []Experience) error {
	klog.V(3).Infof("Starting to send %d experiences for training", len(experiences))
	klog.V(3).Infof("=================== MODEL UPDATE PROCESS STARTING ===================")

	// 输出用于训练的经验信息
	klog.V(3).Infof("Sending %d experiences for training:", len(experiences))
	for i, exp := range experiences {
		klog.V(4).Infof("Experience %d: State(len=%d), Action=%d, Reward=%f, NextState(len=%d), Done=%t",
			i, len(exp.State), exp.Action, exp.Reward, len(exp.NextState), exp.Done)
	}

	// 每64个经验触发一次更新
	minExperiences := 64
	if len(experiences) < minExperiences {
		klog.V(3).Infof("Not enough experiences for training. Have %d, minimum required %d",
			len(experiences), minExperiences)
		// 经验不足时仍然发送，但记录警告信息
		klog.V(3).Infof("Sending experiences anyway to maintain update frequency")
	}

	// 限制单次发送的经验数量以避免内存问题
	maxExperiences := 512
	if len(experiences) > maxExperiences {
		klog.V(3).Infof("Too many experiences (%d) for single training session, limiting to %d", len(experiences), maxExperiences)
		// 只发送最新的经验
		experiences = experiences[len(experiences)-maxExperiences:]
		klog.V(3).Infof("Limited to latest %d experiences", len(experiences))
	}

	// 构造训练数据
	states := make([][]float64, len(experiences))
	actions := make([]int32, len(experiences))
	rewards := make([]float64, len(experiences))
	dones := make([]bool, len(experiences))

	actionDistribution := make(map[int32]int) // 统计动作分布

	for i, exp := range experiences {
		states[i] = exp.State
		actions[i] = exp.Action
		rewards[i] = exp.Reward
		dones[i] = exp.Done
		actionDistribution[exp.Action]++ // 统计动作分布
	}

	// 输出动作分布统计
	actionStats := ""
	for action, count := range actionDistribution {
		actionStats += fmt.Sprintf("Action %d: %d times, ", action, count)
	}
	klog.V(3).Infof("Action distribution: %s", actionStats)

	// 构造请求数据
	updateData := map[string]interface{}{
		"states":  states,
		"actions": actions,
		"rewards": rewards,
		"dones":   dones,
	}

	// 将数据转换为JSON
	jsonData, err := json.Marshal(updateData)
	if err != nil {
		klog.Errorf("Failed to marshal update data: %v", err)
		return fmt.Errorf("failed to marshal update data: %v", err)
	}

	klog.V(4).Infof("Sending training data to PPO service. Data size: %d bytes", len(jsonData))

	// 构造HTTP请求
	url := fmt.Sprintf("http://%s/update", pp.httpAddress)
	klog.V(3).Infof("Sending update request to: %s", url)

	ctx, cancel := context.WithTimeout(context.Background(), pp.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		klog.Errorf("Failed to create HTTP request: %v", err)
		return fmt.Errorf("failed to create HTTP request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// 发送HTTP请求
	client := &http.Client{}
	klog.V(3).Infof("Sending HTTP request to PPO service for model update")
	resp, err := client.Do(req)
	if err != nil {
		klog.Errorf("Failed to send update request: %v", err)
		return fmt.Errorf("failed to send update request: %v", err)
	}
	defer resp.Body.Close()

	klog.V(3).Infof("Received response from PPO service with status: %d", resp.StatusCode)

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		klog.Errorf("PPO service returned error status %d with body: %s", resp.StatusCode, string(body))
		return fmt.Errorf("PPO service returned error status %d", resp.StatusCode)
	}

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		klog.Errorf("Failed to read response body: %v", err)
		return fmt.Errorf("failed to read response body: %v", err)
	}

	klog.V(3).Infof("Model update response: %s", string(body))
	klog.V(3).Infof("=================== MODEL UPDATE PROCESS COMPLETED ===================")

	return nil
}
