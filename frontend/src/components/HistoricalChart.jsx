import React, { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const RANGES = { "1Y": 12, "3Y": 36, "5Y": 60, All: Infinity };

export default function HistoricalChart({ data }) {
  const [range, setRange] = useState("All");

  const chartData = useMemo(() => {
    if (!data || !data.dates) return [];
    const limit = RANGES[range];
    const start = limit === Infinity ? 0 : Math.max(0, data.dates.length - limit);
    return data.dates.slice(start).map((d, i) => ({
      date: d,
      [data.long_label]: data.cumulative_long[start + i],
      [data.short_label]: data.cumulative_short[start + i],
    }));
  }, [data, range]);

  if (!data) return null;

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {Object.keys(RANGES).map((r) => (
          <button
            key={r}
            onClick={() => setRange(r)}
            style={{
              padding: "4px 12px",
              borderRadius: 6,
              border: "1px solid #e2e8f0",
              backgroundColor: range === r ? "#3b82f6" : "#fff",
              color: range === r ? "#fff" : "#64748b",
              cursor: "pointer",
              fontWeight: 500,
              fontSize: 13,
            }}
          >
            {r}
          </button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} tickFormatter={(d) => d.slice(0, 7)} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey={data.long_label} stroke="#3b82f6" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey={data.short_label} stroke="#f59e0b" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
