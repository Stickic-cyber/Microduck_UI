import React, { useEffect, useState } from "react";
import { Download, Pause, Play, RefreshCw, Square } from "lucide-react";
import { exportOnnx, fetchCheckpoints, fetchInferStatus, fetchOnnxModels, startInfer, stopInfer } from "../api";
import { Joystick } from "../components/Joystick";
import { RenderCanvas } from "../components/RenderCanvas";
import { Checkpoint, FrameMessage, OnnxModel } from "../types";

interface InferPanelProps {
  latestFrame: FrameMessage | null;
  inferConnected: boolean;
  onSendCommand: (vx: number, vy: number, wz: number, action?: string) => void;
}

export const InferPanel: React.FC<InferPanelProps> = ({
  latestFrame,
  inferConnected,
  onSendCommand,
}) => {
  const [models, setModels] = useState<OnnxModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isActive, setIsActive] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Export checkpoint modal
  const [showExportModal, setShowExportModal] = useState(false);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [exportFilename, setExportFilename] = useState<string>("");
  const [exporting, setExporting] = useState(false);

  const loadModels = async () => {
    try {
      const list = await fetchOnnxModels();
      setModels(list);
      if (list.length > 0 && !selectedModel) {
        setSelectedModel(list[0].path);
      }
    } catch (e: any) {
      // ignore
    }
  };

  useEffect(() => {
    loadModels();
    fetchInferStatus().then((st) => {
      setIsActive(st.active);
      setIsPaused(st.paused);
      if (st.onnx_path) setSelectedModel(st.onnx_path);
    }).catch(() => {});
  }, []);

  const handleStart = async (modelPath?: string) => {
    const target = modelPath || selectedModel;
    if (!target) return;
    try {
      setLoading(true);
      setError(null);
      await startInfer(target, 25);
      setIsActive(true);
      setIsPaused(false);
    } catch (e: any) {
      setError(e.message || "启动推理失败");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      setLoading(true);
      await stopInfer();
      setIsActive(false);
      setIsPaused(false);
    } catch (e: any) {
      setError(e.message || "停止推理失败");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    onSendCommand(0, 0, 0, "reset");
  };

  const handleTogglePause = () => {
    if (isPaused) {
      onSendCommand(0, 0, 0, "resume");
      setIsPaused(false);
    } else {
      onSendCommand(0, 0, 0, "pause");
      setIsPaused(true);
    }
  };

  const handleOpenExport = async () => {
    try {
      const list = await fetchCheckpoints();
      setCheckpoints(list);
      if (list.length > 0) {
        setSelectedCheckpoint(list[0].checkpoint_path);
        setExportFilename(`exported_${Date.now()}.onnx`);
      }
      setShowExportModal(true);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleExecuteExport = async () => {
    if (!selectedCheckpoint) return;
    try {
      setExporting(true);
      setError(null);
      await exportOnnx("Mjlab-Velocity-Flat-MicroDuck", selectedCheckpoint, exportFilename);
      setShowExportModal(false);
      await loadModels();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white overflow-y-auto p-4 space-y-3">
      {/* 1. MuJoCo Viewport */}
      <RenderCanvas
        frame={latestFrame}
        active={isActive}
        onQuickStart={models.length > 0 ? () => handleStart(models[0].path) : undefined}
      />

      {error && (
        <div className="p-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded">
          {error}
        </div>
      )}

      {/* 2. Model Selector & Controls Bar */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-3">
        <div className="flex items-center space-x-2">
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-700 mb-1">ONNX 策略模型</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isActive}
              className="w-full px-2.5 py-1.5 bg-white border border-gray-300 rounded text-xs font-mono text-gray-800 disabled:bg-gray-100"
            >
              {models.length === 0 && <option value="">未找到 .onnx 模型</option>}
              {models.map((m) => (
                <option key={m.path} value={m.path}>
                  {m.name} ({m.source})
                </option>
              ))}
            </select>
          </div>

          <div className="pt-5">
            <button
              onClick={handleOpenExport}
              className="px-2.5 py-1.5 bg-white hover:bg-gray-100 border border-gray-300 rounded text-xs text-gray-700 flex items-center space-x-1"
              title="从 Checkpoint 导出新 ONNX"
            >
              <Download className="w-3.5 h-3.5" />
              <span>导出 ONNX</span>
            </button>
          </div>
        </div>

        {/* Execution Controls */}
        <div className="flex items-center space-x-2">
          {isActive ? (
            <>
              <button
                onClick={handleStop}
                disabled={loading}
                className="flex-1 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded text-xs font-medium flex items-center justify-center space-x-1.5 transition disabled:opacity-50"
              >
                <Square className="w-3.5 h-3.5" />
                <span>停止推理</span>
              </button>

              <button
                onClick={handleTogglePause}
                className="px-3 py-1.5 bg-white hover:bg-gray-100 border border-gray-300 text-gray-700 rounded text-xs font-medium flex items-center space-x-1"
              >
                {isPaused ? <Play className="w-3.5 h-3.5 text-green-600" /> : <Pause className="w-3.5 h-3.5" />}
                <span>{isPaused ? "继续" : "暂停"}</span>
              </button>

              <button
                onClick={handleReset}
                className="px-3 py-1.5 bg-white hover:bg-gray-100 border border-gray-300 text-gray-700 rounded text-xs font-medium flex items-center space-x-1"
                title="重置仿真姿态到初始状态"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>重置姿态</span>
              </button>
            </>
          ) : (
            <button
              onClick={() => handleStart()}
              disabled={loading || !selectedModel}
              className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-xs font-medium flex items-center justify-center space-x-1.5 transition disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5" />
              <span>启动仿真推理</span>
            </button>
          )}
        </div>
      </div>

      {/* 3. Teleop Joystick */}
      <Joystick onCommand={onSendCommand} disabled={!isActive || isPaused} />

      {/* Export ONNX Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg max-w-lg w-full p-4 space-y-3 shadow-xl">
            <div className="flex items-center justify-between border-b border-gray-200 pb-2">
              <h3 className="text-sm font-semibold text-gray-900">导出 Checkpoint (.pt) 为 ONNX</h3>
              <button onClick={() => setShowExportModal(false)} className="text-gray-400 hover:text-gray-700">
                ✕
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-gray-600 mb-1">选择 Checkpoint</label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded font-mono"
                >
                  {checkpoints.map((c) => (
                    <option key={c.checkpoint_path} value={c.checkpoint_path}>
                      {c.checkpoint_name} ({c.run_name})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-gray-600 mb-1">目标文件名 (.onnx)</label>
                <input
                  type="text"
                  value={exportFilename}
                  onChange={(e) => setExportFilename(e.target.value)}
                  className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded font-mono"
                />
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-2 border-t border-gray-200">
              <button
                onClick={() => setShowExportModal(false)}
                className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs"
              >
                取消
              </button>
              <button
                onClick={handleExecuteExport}
                disabled={exporting || !selectedCheckpoint}
                className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-medium disabled:opacity-50"
              >
                {exporting ? "导出中..." : "开始导出"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
