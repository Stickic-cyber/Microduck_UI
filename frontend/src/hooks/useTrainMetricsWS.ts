import { useEffect, useRef, useState } from "react";
import { MetricPoint, TrainStatus } from "../types";

export function useTrainMetricsWS() {
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [latestMetric, setLatestMetric] = useState<MetricPoint | null>(null);
  const [status, setStatus] = useState<TrainStatus>({ state: "idle" });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);

  useEffect(() => {
    let unmounted = false;

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/ws/train_metrics`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (unmounted) return;
        setConnected(true);
      };

      ws.onmessage = (event) => {
        if (unmounted) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === "status") {
            setStatus({
              state: data.state,
              run_name: data.run_name,
              task_id: data.task_id,
              pid: data.pid,
              log_dir: data.log_dir,
              error_message: data.message || data.error_message,
              current_iteration: data.current_iteration,
              max_iterations: data.max_iterations,
              mean_reward: data.mean_reward,
              eta_str: data.eta_str,
              elapsed_str: data.elapsed_str,
              elapsed_sec: data.elapsed_sec,
              latest_log: data.latest_log,
            });
          } else if (data.type === "metric") {
            setLatestMetric(data);
            setMetrics((prev) => {
              const updated = [...prev, data];
              // Keep maximum 2000 points
              return updated.length > 2000 ? updated.slice(updated.length - 2000) : updated;
            });
          }
        } catch (e) {
          // ignore parsing error
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        setConnected(false);
        reconnectTimeoutRef.current = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      unmounted = true;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const clearMetrics = () => {
    setMetrics([]);
    setLatestMetric(null);
  };

  return { metrics, latestMetric, status, setStatus, connected, clearMetrics };
}
