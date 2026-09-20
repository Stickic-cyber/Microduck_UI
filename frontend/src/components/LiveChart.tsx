import React, { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MetricPoint } from "../types";

interface LiveChartProps {
  metrics: MetricPoint[];
}

export const LiveChart: React.FC<LiveChartProps> = ({ metrics }) => {
  const [showLoss, setShowLoss] = useState(true);
  const [lossKey, setLossKey] = useState("value_loss");

  // Downsample to max 400 points for chart rendering performance
  const chartData = useMemo(() => {
    if (metrics.length <= 400) return metrics;
    const factor = Math.ceil(metrics.length / 400);
    return metrics.filter((_, idx) => idx % factor === 0 || idx === metrics.length - 1);
  }, [metrics]);

  // Extract available loss keys from data
  const availableLossKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const m of metrics.slice(-20)) {
      if (m.loss) {
        Object.keys(m.loss).forEach((k) => keys.add(k));
      }
    }
    return Array.from(keys);
  }, [metrics]);

  const formattedData = useMemo(() => {
    return chartData.map((d) => ({
      iteration: d.iteration,
      mean_reward: d.mean_reward,
      loss_val: d.loss ? d.loss[lossKey] ?? d.loss["loss"] ?? Object.values(d.loss)[0] : undefined,
    }));
  }, [chartData, lossKey]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-2 text-xs">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-blue-600"></span>
            <span className="text-gray-700 font-medium">Reward (左轴)</span>
          </div>

          <label className="flex items-center space-x-1.5 cursor-pointer text-gray-600 hover:text-gray-900">
            <input
              type="checkbox"
              checked={showLoss}
              onChange={(e) => setShowLoss(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-0"
            />
            <span className="w-2.5 h-2.5 rounded-sm bg-orange-600"></span>
            <span>Loss (右轴)</span>
          </label>

          {showLoss && availableLossKeys.length > 0 && (
            <select
              value={lossKey}
              onChange={(e) => setLossKey(e.target.value)}
              className="px-1.5 py-0.5 text-xs bg-white border border-gray-200 rounded text-gray-700 focus:outline-none"
            >
              {availableLossKeys.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="text-gray-400 font-mono">
          采样点数: {chartData.length} / {metrics.length}
        </div>
      </div>

      <div className="flex-1 w-full min-h-[220px]">
        {metrics.length === 0 ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 text-xs border border-dashed border-gray-200 rounded-lg">
            <span>暂无训练指标数据</span>
            <span className="text-[11px] text-gray-400 mt-1">启动训练后将自动在此绘制实时曲线</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={formattedData} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis
                dataKey="iteration"
                tick={{ fontSize: 11, fill: "#6B7280", fontFamily: "monospace" }}
                stroke="#9CA3AF"
              />
              <YAxis
                yAxisId="reward"
                domain={["auto", "auto"]}
                tick={{ fontSize: 11, fill: "#2563EB", fontFamily: "monospace" }}
                stroke="#93C5FD"
              />
              {showLoss && (
                <YAxis
                  yAxisId="loss"
                  orientation="right"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 11, fill: "#EA580C", fontFamily: "monospace" }}
                  stroke="#FDBA74"
                />
              )}
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(255, 255, 255, 0.95)",
                  borderColor: "#E5E7EB",
                  borderRadius: "6px",
                  fontSize: "12px",
                  fontFamily: "monospace",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
                }}
              />
              <Line
                yAxisId="reward"
                type="monotone"
                dataKey="mean_reward"
                name="Reward"
                stroke="#2563EB"
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />
              {showLoss && (
                <Line
                  yAxisId="loss"
                  type="monotone"
                  dataKey="loss_val"
                  name={`Loss (${lossKey})`}
                  stroke="#EA580C"
                  strokeWidth={1.5}
                  strokeDasharray="4 2"
                  dot={false}
                  isAnimationActive={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
