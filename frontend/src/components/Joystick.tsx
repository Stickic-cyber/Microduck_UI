import React, { useCallback, useEffect, useRef, useState } from "react";
import { RotateCcw, RotateCw, Square } from "lucide-react";

interface JoystickProps {
  onCommand: (vx: number, vy: number, wz: number, action?: string) => void;
  disabled?: boolean;
}

export const Joystick: React.FC<JoystickProps> = ({ onCommand, disabled = false }) => {
  const [knobPos, setKnobPos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [wz, setWz] = useState(0);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const commandRef = useRef({ vx: 0, vy: 0, wz: 0 });
  const keysDown = useRef<Set<string>>(new Set());

  const RADIUS = 55; // Joystick radius in pixels
  const MAX_VX = 0.4; // Max forward velocity m/s
  const MAX_VY = 0.3; // Max lateral strafe velocity m/s
  const MAX_WZ = 1.0; // Max turn rate rad/s

  // Throttled command emitter (every 50ms)
  useEffect(() => {
    if (disabled) return;
    const interval = setInterval(() => {
      const { vx, vy, wz } = commandRef.current;
      onCommand(vx, vy, wz);
    }, 50);
    return () => clearInterval(interval);
  }, [disabled, onCommand]);

  const updateVelocity = useCallback((dx: number, dy: number, curWz: number) => {
    // dy: up is negative in screen coordinates -> forward (+vx)
    const normX = Math.max(-1, Math.min(1, dx / RADIUS));
    const normY = Math.max(-1, Math.min(1, -dy / RADIUS));

    const vx = parseFloat((normY * MAX_VX).toFixed(2));
    const vy = parseFloat((normX * MAX_VY).toFixed(2));
    const newWz = parseFloat(curWz.toFixed(2));

    commandRef.current = { vx, vy, wz: newWz };
  }, []);

  // Mouse & Touch handling
  const handlePointerDown = (e: React.PointerEvent) => {
    if (disabled) return;
    setIsDragging(true);
    handlePointerMove(e);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging && e.buttons !== 1) return;
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    let dx = e.clientX - centerX;
    let dy = e.clientY - centerY;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > RADIUS) {
      dx = (dx / dist) * RADIUS;
      dy = (dy / dist) * RADIUS;
    }

    setKnobPos({ x: dx, y: dy });
    updateVelocity(dx, dy, wz);
  };

  const handlePointerUp = () => {
    setIsDragging(false);
    setKnobPos({ x: 0, y: 0 });
    updateVelocity(0, 0, wz);
  };

  // Turn rate adjustments
  const handleTurn = (delta: number) => {
    const nextWz = Math.max(-MAX_WZ, Math.min(MAX_WZ, wz + delta));
    setWz(nextWz);
    updateVelocity(knobPos.x, knobPos.y, nextWz);
  };

  const handleCoast = () => {
    setKnobPos({ x: 0, y: 0 });
    setWz(0);
    commandRef.current = { vx: 0, vy: 0, wz: 0 };
    onCommand(0, 0, 0);
  };

  // Keyboard controls listener (WASD / Arrows)
  useEffect(() => {
    if (disabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't capture when typing in an input form
      if (["INPUT", "SELECT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      const key = e.key.toLowerCase();
      keysDown.current.add(key);

      let vx = 0;
      let vy = 0;
      let keyWz = wz;

      if (keysDown.current.has("w") || keysDown.current.has("arrowup")) vx += MAX_VX;
      if (keysDown.current.has("s") || keysDown.current.has("arrowdown")) vx -= MAX_VX;
      if (keysDown.current.has("a") || keysDown.current.has("arrowleft")) vy -= MAX_VY;
      if (keysDown.current.has("d") || keysDown.current.has("arrowright")) vy += MAX_VY;
      if (keysDown.current.has("q")) keyWz = -0.8;
      if (keysDown.current.has("e")) keyWz = 0.8;
      if (keysDown.current.has(" ")) {
        handleCoast();
        return;
      }

      commandRef.current = { vx, vy, wz: keyWz };
      setKnobPos({
        x: (vy / MAX_VY) * (RADIUS * 0.8),
        y: -(vx / MAX_VX) * (RADIUS * 0.8),
      });
      setWz(keyWz);
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      keysDown.current.delete(key);

      let vx = 0;
      let vy = 0;
      let keyWz = 0;

      if (keysDown.current.has("w") || keysDown.current.has("arrowup")) vx += MAX_VX;
      if (keysDown.current.has("s") || keysDown.current.has("arrowdown")) vx -= MAX_VX;
      if (keysDown.current.has("a") || keysDown.current.has("arrowleft")) vy -= MAX_VY;
      if (keysDown.current.has("d") || keysDown.current.has("arrowright")) vy += MAX_VY;

      commandRef.current = { vx, vy, wz: keyWz };
      setKnobPos({
        x: (vy / MAX_VY) * (RADIUS * 0.8),
        y: -(vx / MAX_VX) * (RADIUS * 0.8),
      });
      setWz(keyWz);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [disabled, wz, updateVelocity]);

  return (
    <div className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg p-3 select-none">
      <div className="w-full flex items-center justify-between text-xs text-gray-500 mb-2">
        <span className="font-medium text-gray-700">遥控交互</span>
        <div className="flex items-center space-x-2 font-mono text-[11px]">
          <span>vx: {commandRef.current.vx > 0 ? `+${commandRef.current.vx}` : commandRef.current.vx}</span>
          <span>vy: {commandRef.current.vy > 0 ? `+${commandRef.current.vy}` : commandRef.current.vy}</span>
          <span>wz: {commandRef.current.wz > 0 ? `+${commandRef.current.wz}` : commandRef.current.wz}</span>
        </div>
      </div>

      <div className="flex items-center justify-around w-full">
        {/* Virtual Joystick Touch Pad */}
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          className={`relative w-28 h-28 rounded-full border-2 border-dashed border-gray-300 bg-white flex items-center justify-center cursor-grab active:cursor-grabbing shadow-sm ${
            disabled ? "opacity-50 pointer-events-none" : ""
          }`}
        >
          {/* Center Crosshairs */}
          <div className="absolute w-full h-px bg-gray-200"></div>
          <div className="absolute h-full w-px bg-gray-200"></div>

          {/* Knob */}
          <div
            style={{
              transform: `translate(${knobPos.x}px, ${knobPos.y}px)`,
              transition: isDragging ? "none" : "transform 0.15s ease-out",
            }}
            className="w-10 h-10 rounded-full bg-blue-600 shadow-md border-2 border-white flex items-center justify-center pointer-events-none"
          >
            <div className="w-2 h-2 rounded-full bg-white"></div>
          </div>
        </div>

        {/* Turn Controls & Coast Button */}
        <div className="flex flex-col space-y-2">
          <div className="flex items-center space-x-1.5">
            <button
              onClick={() => handleTurn(-0.4)}
              disabled={disabled}
              className="p-2 bg-white hover:bg-gray-100 border border-gray-200 rounded text-gray-700 disabled:opacity-50 active:bg-gray-200"
              title="向左自转 (Q)"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={handleCoast}
              disabled={disabled}
              className="px-2.5 py-1.5 bg-white hover:bg-gray-100 border border-gray-200 rounded text-xs font-medium text-gray-700 disabled:opacity-50 active:bg-gray-200"
              title="回中刹停 (Space)"
            >
              回中
            </button>
            <button
              onClick={() => handleTurn(0.4)}
              disabled={disabled}
              className="p-2 bg-white hover:bg-gray-100 border border-gray-200 rounded text-gray-700 disabled:opacity-50 active:bg-gray-200"
              title="向右自转 (E)"
            >
              <RotateCw className="w-4 h-4" />
            </button>
          </div>

          <div className="text-[11px] text-gray-500 text-center font-mono">
            WASD 移动 / QE 自转 / 空格 刹停
          </div>
        </div>
      </div>
    </div>
  );
};
