import React from "react";
import { FrameMessage } from "../types";
import { Play } from "lucide-react";

interface RenderCanvasProps {
  frame: FrameMessage | null;
  active: boolean;
  onQuickStart?: () => void;
}

export const RenderCanvas: React.FC<RenderCanvasProps> = ({ frame, active, onQuickStart }) => {
  return (
    <div className="relative w-full aspect-[4/3] bg-gray-900 rounded-lg overflow-hidden flex items-center justify-center border border-gray-200 shadow-inner">
      {active && frame?.jpeg_b64 ? (
        <>
          <img
            src={`data:image/jpeg;base64,${frame.jpeg_b64}`}
            alt="MuJoCo Offscreen Stream"
            className="w-full h-full object-contain"
          />

          {/* Telemetry HUD Overlay */}
          <div className="absolute top-2 left-2 bg-black/60 text-white px-2.5 py-1 rounded text-xs font-mono space-y-0.5 backdrop-blur-sm">
            <div className="flex items-center space-x-3">
              <span>时间: {frame.sim_time.toFixed(2)}s</span>
              <span>步数: {frame.step}</span>
            </div>
            <div className="text-[11px] text-gray-300">
              指令: vx={frame.cmd.vx.toFixed(2)} vy={frame.cmd.vy.toFixed(2)} wz={frame.cmd.wz.toFixed(2)}
            </div>
          </div>

          <div className="absolute top-2 right-2 bg-black/60 text-green-400 px-2 py-0.5 rounded text-[11px] font-mono flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
            <span>25 FPS</span>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center justify-center text-center p-6 text-gray-400">
          <div className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center mb-3 text-gray-500">
            <Play className="w-5 h-5 ml-0.5" />
          </div>
          <p className="text-sm font-medium text-gray-300 mb-1">推理可视化未启动</p>
          <p className="text-xs text-gray-500 max-w-xs mb-4">
            在下方选择已导出的 ONNX 模型策略，点击"启动推理"即可查看 MuJoCo 离屏画面与实时遥控。
          </p>
          {onQuickStart && (
            <button
              onClick={onQuickStart}
              className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded font-medium transition"
            >
              加载默认模型启动
            </button>
          )}
        </div>
      )}
    </div>
  );
};
