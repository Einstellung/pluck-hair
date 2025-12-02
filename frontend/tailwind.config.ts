import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: "#0f172a",
        panel: "#1f2937",
        border: "#243047",
        accent: "#16a34a",
        danger: "#dc2626",
        text: "#e5e7eb",
        muted: "#9ca3af",
      },
      borderRadius: {
        lg: "14px",
        md: "10px",
        sm: "8px",
      },
      boxShadow: {
        panel: "0 10px 40px rgba(0,0,0,0.25)",
      },
      fontFamily: {
        sans: [
          "\"Noto Sans SC\"",
          "\"PingFang SC\"",
          "\"Microsoft YaHei\"",
          "Arial",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
export default config;
