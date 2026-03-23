import React from "react";

function daColor(value) {
  if (value >= 0.55) return "#22c55e";
  if (value < 0.50) return "#ef4444";
  return "#64748b";
}

export default function ModelMetrics({ metrics }) {
  if (!metrics || metrics.length === 0) return null;

  return (
    <div style={{ backgroundColor: "#fff", borderRadius: 8, border: "1px solid #e2e8f0", padding: 20 }}>
      <h3 style={{ margin: "0 0 12px", fontSize: 16, color: "#1e293b" }}>Model Backtest Metrics</h3>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #e2e8f0" }}>
              <th style={{ textAlign: "left", padding: "8px 12px", color: "#64748b" }}>Comparison</th>
              <th style={{ textAlign: "left", padding: "8px 12px", color: "#64748b" }}>Horizon</th>
              <th style={{ textAlign: "right", padding: "8px 12px", color: "#64748b" }}>MAE</th>
              <th style={{ textAlign: "right", padding: "8px 12px", color: "#64748b" }}>RMSE</th>
              <th style={{ textAlign: "right", padding: "8px 12px", color: "#64748b" }}>Dir. Accuracy</th>
              <th style={{ textAlign: "right", padding: "8px 12px", color: "#64748b" }}>IC</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #f1f5f9" }}>
                <td style={{ padding: "8px 12px" }}>{m.comparisonLabel}</td>
                <td style={{ padding: "8px 12px" }}>{m.horizonLabel}</td>
                <td style={{ padding: "8px 12px", textAlign: "right" }}>
                  {m.mae != null ? m.mae.toFixed(4) : "—"}
                </td>
                <td style={{ padding: "8px 12px", textAlign: "right" }}>
                  {m.rmse != null ? m.rmse.toFixed(4) : "—"}
                </td>
                <td style={{
                  padding: "8px 12px",
                  textAlign: "right",
                  fontWeight: 600,
                  color: daColor(m.directional_accuracy),
                }}>
                  {m.directional_accuracy != null ? (m.directional_accuracy * 100).toFixed(1) + "%" : "—"}
                </td>
                <td style={{ padding: "8px 12px", textAlign: "right" }}>
                  {m.ic != null ? m.ic.toFixed(3) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
