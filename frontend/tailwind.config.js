/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:       "#0a0c0f",
        card:     "#111318",
        elevated: "#181c22",
        border:   "#1e2530",
        green:    "#22d48a",
        amber:    "#f0a832",
        blue:     "#4d9ef5",
        coral:    "#f06060",
        purple:   "#9b7ef5",
      },
      fontFamily: {
        display: ["Syne", "sans-serif"],
        body:    ["Inter", "sans-serif"],
        mono:    ["DM Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
