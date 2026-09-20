import React from "react";
import { Header } from "./components/Header";
import { InferPanel } from "./panels/InferPanel";
import { TrainPanel } from "./panels/TrainPanel";
import { useRenderStreamWS } from "./hooks/useRenderStreamWS";
import { useTrainMetricsWS } from "./hooks/useTrainMetricsWS";

export function App() {
  const {
    metrics,
    latestMetric,
    status: trainStatus,
    connected: trainConnected,
    clearMetrics,
  } = useTrainMetricsWS();

  const {
    latestFrame,
    connected: inferConnected,
    sendCommand,
  } = useRenderStreamWS();

  return (
    <div className="flex flex-col h-screen w-screen bg-[#F7F7F8] overflow-hidden">
      <Header trainConnected={trainConnected} inferConnected={inferConnected} />

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 overflow-hidden">
        {/* Left: RL Training & Monitoring */}
        <div className="h-full overflow-hidden">
          <TrainPanel
            status={trainStatus}
            metrics={metrics}
            latestMetric={latestMetric}
            onClearMetrics={clearMetrics}
          />
        </div>

        {/* Right: ONNX Inference & Teleoperation */}
        <div className="h-full overflow-hidden">
          <InferPanel
            latestFrame={latestFrame}
            inferConnected={inferConnected}
            onSendCommand={sendCommand}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
