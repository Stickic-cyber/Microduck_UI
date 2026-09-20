import React from "react";
import { Activity, Wifi, WifiOff } from "lucide-react";

interface HeaderProps {
  trainConnected: boolean;
  inferConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({ trainConnected, inferConnected }) => {
  return (
    <header className="h-14 bg-white border-b border-gray-200 px-6 flex items-center justify-between select-none">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
          MD
        </div>
        <div>
          <h1 className="text-base font-semibold text-gray-900 tracking-tight flex items-center space-x-2">
            <span>Microduck RL WebUI</span>
            <span className="text-xs font-normal px-2 py-0.5 rounded bg-gray-100 text-gray-600 border border-gray-200">
              v1.0.0
            </span>
          </h1>
        </div>
      </div>

      <div className="flex items-center space-x-5 text-xs text-gray-600">
        <div className="flex items-center space-x-1.5">
          <span className="text-gray-400">训练流:</span>
          {trainConnected ? (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-green-50 text-green-700 font-mono font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-green-600 mr-1 animate-pulse"></span>
              已连接
            </span>
          ) : (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-400 mr-1"></span>
              离线
            </span>
          )}
        </div>

        <div className="flex items-center space-x-1.5">
          <span className="text-gray-400">渲染流:</span>
          {inferConnected ? (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-green-50 text-green-700 font-mono font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-green-600 mr-1 animate-pulse"></span>
              已连接
            </span>
          ) : (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-400 mr-1"></span>
              待命
            </span>
          )}
        </div>

        <div className="h-4 w-px bg-gray-200"></div>

        <div className="text-gray-500 font-mono">
          MuJoCo Warp + RSL-RL PPO
        </div>
      </div>
    </header>
  );
};
