import { useCallback, useEffect, useRef, useState } from "react";
import { FrameMessage } from "../types";

export function useRenderStreamWS() {
  const [latestFrame, setLatestFrame] = useState<FrameMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);

  useEffect(() => {
    let unmounted = false;

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/ws/render_stream`;

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
          if (data.type === "frame") {
            setLatestFrame(data);
          }
        } catch (e) {
          // ignore
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

  const sendCommand = useCallback((vx: number, vy: number, wz: number, action?: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (action) {
        wsRef.current.send(JSON.stringify({ type: "action", action }));
      }
      wsRef.current.send(JSON.stringify({ type: "command", vx, vy, wz }));
    }
  }, []);

  return { latestFrame, connected, sendCommand };
}
