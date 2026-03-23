import React from "react";
import PredictionCard from "./PredictionCard";
import HistoricalChart from "./HistoricalChart";

const HORIZON_ORDER = ["1m", "3m", "6m", "12m"];

export default function ComparisonPanel({ comparisonKey, label, predictions, historicalData }) {
  const sorted = HORIZON_ORDER
    .map((h) => predictions.find((p) => p.horizon === h))
    .filter(Boolean);

  return (
    <div style={{ marginBottom: 32 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: "#1e293b", marginBottom: 16 }}>{label}</h2>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        gap: 12,
        marginBottom: 20,
      }}>
        {sorted.map((p) => (
          <PredictionCard
            key={p.horizon}
            horizon={p.horizon}
            horizonLabel={p.horizonLabel}
            prediction={p.prediction}
            confidence={p.confidence}
            recommendation={p.recommendation}
            direction={p.direction}
            magnitude={p.magnitude}
          />
        ))}
      </div>

      <div style={{ backgroundColor: "#fff", borderRadius: 8, border: "1px solid #e2e8f0", padding: 20 }}>
        <h3 style={{ margin: "0 0 12px", fontSize: 16, color: "#1e293b" }}>Cumulative Performance</h3>
        <HistoricalChart data={historicalData} />
      </div>
    </div>
  );
}
