import React from "react";

const confidenceColors = {
  high: "#22c55e",
  medium: "#f59e0b",
  low: "#94a3b8",
};

const magnitudeBarWidth = {
  negligible: "10%",
  modest: "35%",
  moderate: "60%",
  strong: "90%",
};

export default function PredictionCard({ horizon, horizonLabel, prediction, confidence, recommendation, direction, magnitude }) {
  const predPct = (prediction * 100).toFixed(2);
  const isPositive = prediction > 0;
  const color = isPositive ? "#22c55e" : "#ef4444";

  return (
    <div style={{
      border: "1px solid #e2e8f0",
      borderLeft: `4px solid ${color}`,
      borderRadius: 8,
      padding: "16px",
      backgroundColor: "#fff",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 14, color: "#64748b" }}>{horizonLabel}</span>
        <span style={{
          fontSize: 12,
          padding: "2px 8px",
          borderRadius: 12,
          backgroundColor: confidenceColors[confidence] + "20",
          color: confidenceColors[confidence],
          fontWeight: 600,
        }}>
          {confidence}
        </span>
      </div>

      <div style={{ fontSize: 28, fontWeight: 700, color, marginBottom: 4 }}>
        {isPositive ? "+" : ""}{predPct}%
      </div>

      <div style={{ fontSize: 13, color: "#475569", marginBottom: 12 }}>
        {recommendation}
      </div>

      <div style={{ backgroundColor: "#f1f5f9", borderRadius: 4, height: 6, overflow: "hidden" }}>
        <div style={{
          width: magnitudeBarWidth[magnitude] || "50%",
          height: "100%",
          backgroundColor: color,
          borderRadius: 4,
          transition: "width 0.3s ease",
        }} />
      </div>
    </div>
  );
}
