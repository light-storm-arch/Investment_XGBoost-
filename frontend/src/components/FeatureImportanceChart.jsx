import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { fetchFeatureImportance } from "../api/client";

const COMPARISONS = [
  { key: "value_growth", label: "Value vs Growth" },
  { key: "us_intl", label: "US vs International" },
];
const HORIZONS = ["1m", "3m", "6m", "12m"];

export default function FeatureImportanceChart() {
  const [comparison, setComparison] = useState("value_growth");
  const [horizon, setHorizon] = useState("6m");
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchFeatureImportance(comparison, horizon)
      .then(setData)
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, [comparison, horizon]);

  return (
    <div style={{ backgroundColor: "#fff", borderRadius: 8, border: "1px solid #e2e8f0", padding: 20 }}>
      <h3 style={{ margin: "0 0 12px", fontSize: 16, color: "#1e293b" }}>Feature Importance</h3>

      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <select
          value={comparison}
          onChange={(e) => setComparison(e.target.value)}
          style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #cbd5e1", fontSize: 13 }}
        >
          {COMPARISONS.map((c) => (
            <option key={c.key} value={c.key}>{c.label}</option>
          ))}
        </select>
        <select
          value={horizon}
          onChange={(e) => setHorizon(e.target.value)}
          style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #cbd5e1", fontSize: 13 }}
        >
          {HORIZONS.map((h) => (
            <option key={h} value={h}>{h}</option>
          ))}
        </select>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 40, color: "#94a3b8" }}>Loading...</div>
      ) : (
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 28)}>
          <BarChart data={data} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
