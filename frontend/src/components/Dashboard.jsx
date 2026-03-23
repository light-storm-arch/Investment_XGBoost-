import React from "react";
import ComparisonPanel from "./ComparisonPanel";
import FeatureImportanceChart from "./FeatureImportanceChart";
import ModelMetrics from "./ModelMetrics";

const COMPARISONS = [
  { key: "value_growth", label: "Value vs Growth (VVIAX vs VIGAX)" },
  { key: "us_intl", label: "US vs International (VTSAX vs VTIAX)" },
];

export default function Dashboard({ predictions, historicalData, metrics, onRetrain, retraining }) {
  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: "#0f172a", margin: 0 }}>
            Investment Comparison
          </h1>
          <p style={{ color: "#64748b", margin: "4px 0 0", fontSize: 14 }}>
            XGBoost-powered fund pair predictions
          </p>
        </div>
        <button
          onClick={onRetrain}
          disabled={retraining}
          style={{
            padding: "10px 20px",
            borderRadius: 8,
            border: "none",
            backgroundColor: retraining ? "#94a3b8" : "#3b82f6",
            color: "#fff",
            fontWeight: 600,
            cursor: retraining ? "default" : "pointer",
            fontSize: 14,
          }}
        >
          {retraining ? "Retraining..." : "Retrain Models"}
        </button>
      </div>

      {COMPARISONS.map((c) => (
        <ComparisonPanel
          key={c.key}
          comparisonKey={c.key}
          label={c.label}
          predictions={predictions.filter((p) => p.comparison === c.key)}
          historicalData={historicalData[c.key]}
        />
      ))}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 8 }}>
        <FeatureImportanceChart />
        <ModelMetrics metrics={metrics} />
      </div>
    </div>
  );
}
