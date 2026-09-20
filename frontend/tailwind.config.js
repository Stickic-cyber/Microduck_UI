/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          subtle: "#F7F7F8",
          card: "#FFFFFF",
          border: "#E5E7EB",
        },
        primary: {
          DEFAULT: "#2563EB",
          hover: "#1D4ED8",
          subtle: "#EFF6FF",
        },
        status: {
          success: "#16A34A",
          warning: "#EA580C",
          error: "#DC2626",
          idle: "#6B7280",
        }
      },
      fontFamily: {
        mono: ["Consolas", "Monaco", "Courier New", "monospace"],
      },
    },
  },
  plugins: [],
}
