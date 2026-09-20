import React, { useEffect, useState } from "react";
import { ChevronDown, ChevronUp, FileText, History, Play, Square } from "lucide-react";
import { fetchEnvs, fetchTrainHistory, fetchTrainLogs, startTrain, stopTrain } from "../api";
import { LiveChart } from "../components/LiveChart";
import { MetricPoint, TrainConfigForm, TrainStatus } from "../types";

interface TrainPanelProps {
  status: TrainStatus;
  metrics: MetricPoint[];
  latestMetric: MetricPoint | null;
  onClearMetrics: () => void;
}

export const TrainPanel: React.FC<TrainPanelProps> = ({
  status,
  metrics,
  latestMetric,
  onClearMetrics,
}) => {
  const [envs, setEnvs] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [form, setForm] = useState<TrainConfigForm>({
    env_id: "Mjlab-Velocity-Flat-MicroDuck",
    num_envs: 16,
    max_iterations: 100,
    run_name: "",
    hidden_dims: "512, 256, 128",
    activation: "elu",
    learning_rate: 0.001,
    gamma: 0.99,
    entropy_coef: 0.01,
    clip_param: 0.2,
    value_loss_coef: 1.0,
  });

  useEffect(() => {
    fetchEnvs().then(setEnvs).catch(() => {});
  }, []);

  // Poll training stdout/stderr logs every 1.5s when logs modal is open
  useEffect(() => {
    if (!showLogs) return;
    const updateLogs = () => {
      fetchTrainLogs()
        .then((data) => setLogs(data.logs))
        .catch(() => {});
    };
    updateLogs();
    const interval = setInterval(updateLogs, 1500);
    return () => clearInterval(interval);
  }, [showLogs]);

  const [localElapsed, setLocalElapsed] = useState<number>(0);

  const isRunning = status.state === "running";

  // Real-time ticking stopwatch for elapsed time
  useEffect(() => {
    let interval: any = null;
    if (isRunning) {
      if (typeof status.elapsed_sec === "number" && status.elapsed_sec > 0) {
        setLocalElapsed(status.elapsed_sec);
      }
      interval = setInterval(() => {
        setLocalElapsed((prev) => prev + 1);
      }, 1000);
    } else {
      if (status.state === "idle") {
        setLocalElapsed(0);
      }
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning, status.state]);

  // Sync with server-reported elapsed_sec when available
  useEffect(() => {
    if (typeof status.elapsed_sec === "number" && status.elapsed_sec > 0) {
      setLocalElapsed(status.elapsed_sec);
    }
  }, [status.elapsed_sec]);

  const handleStart = async () => {
    try {
      setLoading(true);
      setError(null);
      setLocalElapsed(0);
      onClearMetrics();
      await startTrain(form);
    } catch (err: any) {
      setError(err.message || "启动训练失败");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!window.confirm("确定要停止当前的训练任务吗？")) return;
    try {
      setLoading(true);
      await stopTrain();
    } catch (err: any) {
      setError(err.message || "停止训练失败");
    } finally {
      setLoading(false);
    }
  };

  const handleOpenLogs = async () => {
    try {
      const data = await fetchTrainLogs();
      setLogs(data.logs);
      setShowLogs(true);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleOpenHistory = async () => {
    try {
      const data = await fetchTrainHistory();
      setHistory(data);
      setShowHistory(true);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const currentIter = status.current_iteration ?? latestMetric?.iteration ?? 0;
  const maxIter = status.max_iterations ?? form.max_iterations ?? 100;
  const progressPercent = maxIter > 0 ? Math.min(100, Math.round((currentIter / maxIter) * 100)) : 0;

  const displayReward =
    latestMetric?.mean_reward !== undefined
      ? latestMetric.mean_reward.toFixed(2)
      : typeof status.mean_reward === "number"
      ? status.mean_reward.toFixed(2)
      : "--";

  const formatSeconds = (sec?: number) => {
    if (sec === undefined || sec === null || isNaN(sec) || sec < 0) return "--";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}m ${s < 10 ? "0" : ""}${s}s`;
  };

  const displayElapsed =
    status.elapsed_str && status.elapsed_str.trim() !== ""
      ? status.elapsed_str
      : typeof status.elapsed_sec === "number" && status.elapsed_sec > 0
      ? formatSeconds(status.elapsed_sec)
      : isRunning
      ? formatSeconds(localElapsed)
      : "--";

  const displayEta =
    status.eta_str && status.eta_str.trim() !== ""
      ? status.eta_str
      : latestMetric?.eta_s !== undefined && latestMetric.eta_s > 0
      ? formatSeconds(latestMetric.eta_s)
      : isRunning && currentIter > 0
      ? "计算中..."
      : "--";

  return (
    <div className="flex flex-col h-full bg-white border-r border-gray-200 overflow-y-auto">
      {/* 1. Header & Telemetry Stat Cards */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isRunning
                  ? "bg-green-500 animate-pulse"
                  : status.state === "finished"
                  ? "bg-blue-500"
                  : status.state === "error"
                  ? "bg-red-500"
                  : status.state === "stopped"
                  ? "bg-orange-500"
                  : "bg-gray-400"
              }`}
            ></span>
            <span className="text-sm font-semibold text-gray-900">
              {isRunning
                ? "训练中"
                : status.state === "finished"
                ? "训练已完成"
                : status.state === "error"
                ? "训练异常"
                : status.state === "stopped"
                ? "已停止"
                : "空闲"}
            </span>
            {status.run_name && (
              <span className="text-xs font-mono text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">
                {status.run_name}
              </span>
            )}
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleOpenLogs}
              className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded text-xs flex items-center space-x-1"
              title="查看训练控制台日志"
            >
              <FileText className="w-4 h-4" />
              <span>日志</span>
            </button>
            <button
              onClick={handleOpenHistory}
              className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded text-xs flex items-center space-x-1"
              title="查看历史训练记录"
            >
              <History className="w-4 h-4" />
              <span>历史</span>
            </button>
          </div>
        </div>

        {/* --- Visual Progress Bar & Phase Info --- */}
        <div className="mb-3 bg-gray-50/90 p-2.5 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <div className="flex items-center space-x-1.5">
              <span className="font-semibold text-gray-700">训练总进度</span>
              {isRunning && (
                <span className="inline-flex items-center px-1.5 py-0.2 rounded text-[10px] font-medium bg-blue-100 text-blue-800 animate-pulse">
                  运行中
                </span>
              )}
            </div>
            <div className="font-mono text-xs">
              <span className="font-bold text-blue-600">{progressPercent}%</span>
              <span className="text-gray-400 ml-1.5">({currentIter} / {maxIter} 轮)</span>
            </div>
          </div>

          {/* Progress track & fill */}
          <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
            <div
              className={`h-2.5 rounded-full transition-all duration-300 ease-out ${
                isRunning
                  ? "bg-gradient-to-r from-blue-500 to-indigo-600"
                  : status.state === "finished"
                  ? "bg-green-500"
                  : status.state === "error"
                  ? "bg-red-500"
                  : "bg-gray-400"
              }`}
              style={{
                width: `${isRunning && progressPercent === 0 ? 3 : progressPercent}%`,
              }}
            />
          </div>

          {/* Dynamic Phase / Status description */}
          <div className="mt-2 flex items-center justify-between text-[11px] text-gray-500">
            <div className="truncate pr-2 font-mono">
              {isRunning ? (
                status.latest_log ? (
                  <span className="text-gray-700 font-medium">{status.latest_log}</span>
                ) : currentIter === 0 ? (
                  <span className="text-blue-600 animate-pulse">⚙️ 正在启动 MuJoCo Warp 并行环境与策略网络...</span>
                ) : (
                  <span>⚡ 正在执行策略优化与梯度迭代...</span>
                )
              ) : status.state === "finished" ? (
                <span className="text-green-600">✅ 训练已完成，ONNX 策略模型就绪</span>
              ) : status.state === "error" ? (
                <span className="text-red-500">❌ 任务异常终止，请点击右上角「日志」排查</span>
              ) : status.state === "stopped" ? (
                <span>⏹️ 训练已手动终止</span>
              ) : (
                <span>就绪，点击下方按钮启动训练</span>
              )}
            </div>
            {latestMetric?.fps !== undefined && latestMetric.fps > 0 && (
              <span className="shrink-0 font-mono text-[10px] text-gray-500 bg-white px-1.5 py-0.5 rounded border border-gray-200">
                {latestMetric.fps.toFixed(0)} FPS
              </span>
            )}
          </div>
        </div>

        {/* Quick Numbers Bar */}
        <div className="grid grid-cols-4 gap-2 text-center bg-gray-50 p-2.5 rounded-lg border border-gray-200">
          <div>
            <div className="text-[11px] text-gray-500">迭代轮数</div>
            <div className="text-base font-bold font-mono text-gray-900">
              {currentIter} <span className="text-xs font-normal text-gray-400">/ {maxIter}</span>
            </div>
          </div>
          <div>
            <div className="text-[11px] text-gray-500">Reward</div>
            <div className="text-base font-bold font-mono text-blue-600">{displayReward}</div>
          </div>
          <div>
            <div className="text-[11px] text-gray-500">已耗时</div>
            <div className="text-xs font-bold font-mono text-gray-800 mt-1">
              {displayElapsed}
            </div>
          </div>
          <div>
            <div className="text-[11px] text-gray-500">预计剩余 (ETA)</div>
            <div className="text-xs font-bold font-mono text-gray-800 mt-1">
              {displayEta}
            </div>
          </div>
        </div>

        {status.state === "error" && (
          <div className="mt-2.5 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700 flex items-center justify-between">
            <span className="truncate pr-2">训练异常: {status.error_message || "任务非正常退出，请打开终端日志查看"}</span>
            <button
              onClick={handleOpenLogs}
              className="shrink-0 text-red-800 underline font-medium hover:text-red-950"
            >
              查看日志
            </button>
          </div>
        )}

        {status.state === "finished" && (
          <div className="mt-2.5 p-2 bg-green-50 border border-green-200 rounded text-xs text-green-800 flex items-center justify-between">
            <span>🎉 训练已成功完成！模型已自动记录并导出。</span>
            <button
              onClick={handleOpenHistory}
              className="shrink-0 text-green-900 underline font-medium hover:text-green-950"
            >
              查看记录
            </button>
          </div>
        )}
      </div>

      {/* 2. Realtime Chart View */}
      <div className="p-4 border-b border-gray-200 flex-1 min-h-[280px]">
        <LiveChart metrics={metrics} />
      </div>

      {/* 3. Training Form & Controls */}
      <div className="p-4 bg-gray-50 space-y-3">
        {error && (
          <div className="p-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded">
            {error}
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <label className="block text-gray-600 mb-1">环境选择 (Env ID)</label>
            <select
              value={form.env_id}
              disabled={isRunning}
              onChange={(e) => setForm({ ...form, env_id: e.target.value })}
              className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded font-mono text-xs focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
            >
              {envs.map((env) => (
                <option key={env} value={env}>
                  {env}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-gray-600 mb-1">最大迭代轮数 (Iterations)</label>
            <input
              type="number"
              value={form.max_iterations}
              disabled={isRunning}
              onChange={(e) => setForm({ ...form, max_iterations: parseInt(e.target.value, 10) || 100 })}
              className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded font-mono text-xs disabled:bg-gray-100"
            />
          </div>

          <div>
            <label className="block text-gray-600 mb-1">并行环境数 (num_envs)</label>
            <input
              type="number"
              value={form.num_envs}
              disabled={isRunning}
              onChange={(e) => setForm({ ...form, num_envs: parseInt(e.target.value, 10) || 16 })}
              className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded font-mono text-xs disabled:bg-gray-100"
            />
          </div>

          <div>
            <label className="block text-gray-600 mb-1">运行名称 (可选)</label>
            <input
              type="text"
              placeholder="例如: test_run_1"
              value={form.run_name}
              disabled={isRunning}
              onChange={(e) => setForm({ ...form, run_name: e.target.value })}
              className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded text-xs disabled:bg-gray-100"
            />
          </div>
        </div>

        {/* Collapsible Advanced Network & PPO Settings */}
        <div className="border border-gray-200 bg-white rounded">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full px-3 py-2 text-xs font-medium text-gray-600 flex items-center justify-between hover:bg-gray-50"
          >
            <span>高级网络架构 (MLP & PPO)</span>
            {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {showAdvanced && (
            <div className="p-3 border-t border-gray-200 grid grid-cols-2 gap-3 text-xs bg-gray-50/50">
              <div>
                <label className="block text-gray-600 mb-1">隐藏层维度 (hidden_dims)</label>
                <input
                  type="text"
                  value={form.hidden_dims}
                  disabled={isRunning}
                  onChange={(e) => setForm({ ...form, hidden_dims: e.target.value })}
                  className="w-full px-2 py-1 bg-white border border-gray-300 rounded font-mono text-xs"
                />
              </div>

              <div>
                <label className="block text-gray-600 mb-1">激活函数 (activation)</label>
                <select
                  value={form.activation}
                  disabled={isRunning}
                  onChange={(e) => setForm({ ...form, activation: e.target.value })}
                  className="w-full px-2 py-1 bg-white border border-gray-300 rounded text-xs"
                >
                  <option value="elu">ELU</option>
                  <option value="relu">ReLU</option>
                  <option value="tanh">Tanh</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-600 mb-1">学习率 (learning_rate)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={form.learning_rate}
                  disabled={isRunning}
                  onChange={(e) => setForm({ ...form, learning_rate: parseFloat(e.target.value) || 0.001 })}
                  className="w-full px-2 py-1 bg-white border border-gray-300 rounded font-mono text-xs"
                />
              </div>

              <div>
                <label className="block text-gray-600 mb-1">折扣因子 (gamma)</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.gamma}
                  disabled={isRunning}
                  onChange={(e) => setForm({ ...form, gamma: parseFloat(e.target.value) || 0.99 })}
                  className="w-full px-2 py-1 bg-white border border-gray-300 rounded font-mono text-xs"
                />
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2 pt-1">
          {isRunning ? (
            <button
              type="button"
              onClick={handleStop}
              disabled={loading}
              className="flex-1 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-xs font-medium flex items-center justify-center space-x-1.5 transition disabled:opacity-50"
            >
              <Square className="w-3.5 h-3.5" />
              <span>停止当前训练</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={handleStart}
              disabled={loading}
              className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-medium flex items-center justify-center space-x-1.5 transition disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5" />
              <span>启动强化学习训练</span>
            </button>
          )}
        </div>
      </div>

      {/* Logs Modal with Live Polling */}
      {showLogs && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] flex flex-col shadow-xl">
            <div className="p-3 border-b border-gray-200 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <h3 className="text-sm font-semibold text-gray-900">训练终端输出日志</h3>
                <span className="text-[11px] text-green-600 flex items-center">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500 mr-1 animate-pulse"></span>
                  每 1.5s 自动刷新
                </span>
              </div>
              <button
                onClick={() => setShowLogs(false)}
                className="text-gray-400 hover:text-gray-700 text-sm font-bold"
              >
                &times;
              </button>
            </div>
            <div className="p-3 flex-1 overflow-y-auto bg-gray-900 text-gray-200 font-mono text-xs space-y-1">
              {logs.length === 0 ? (
                <div className="text-gray-500">暂无日志输出...</div>
              ) : (
                logs.map((l, i) => <div key={i}>{l}</div>)
              )}
            </div>
            <div className="p-2.5 border-t border-gray-200 flex justify-end">
              <button
                onClick={() => setShowLogs(false)}
                className="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs"
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* History Modal */}
      {showHistory && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg max-w-3xl w-full max-h-[80vh] flex flex-col shadow-xl">
            <div className="p-3 border-b border-gray-200 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-900">历史训练记录</h3>
              <button
                onClick={() => setShowHistory(false)}
                className="text-gray-400 hover:text-gray-700 text-sm font-bold"
              >
                &times;
              </button>
            </div>
            <div className="p-3 flex-1 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500">
                    <th className="pb-2">名称</th>
                    <th className="pb-2">环境</th>
                    <th className="pb-2">状态</th>
                    <th className="pb-2">创建时间</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 font-mono">
                  {history.map((h) => (
                    <tr key={h.id}>
                      <td className="py-2 text-gray-900 font-semibold">{h.run_name}</td>
                      <td className="py-2 text-gray-600">{h.task_id}</td>
                      <td className="py-2">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] ${
                            h.status === "finished"
                              ? "bg-green-100 text-green-800"
                              : h.status === "running"
                              ? "bg-blue-100 text-blue-800"
                              : "bg-gray-100 text-gray-800"
                          }`}
                        >
                          {h.status}
                        </span>
                      </td>
                      <td className="py-2 text-gray-500 text-[11px]">{h.created_at.slice(0, 19).replace("T", " ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
