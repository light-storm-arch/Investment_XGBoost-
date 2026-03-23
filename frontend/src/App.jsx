import React, { useState, useEffect } from "react";
import Dashboard from "./components/Dashboard";
import { fetchPredictions, fetchHistorical, fetchModelMetrics, triggerRetrain } from "./api/client";

export default function App() {
  const [predictions, setPredictions] = useState([]);
  const [historicalData, setHistoricalData] = useState({});
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retraining, setRetraining] = useState(false);

  const loadData = async () => {
    try {
      const [preds, vgHist, uiHist, mets] = await Promise.all([
        fetchPredictions(),
        fetchHistorical("value_growth"),
        fetchHistorical("us_intl"),
        fetchModelMetrics(),
      ]);
      setPredictions(preds);
      setHistoricalData({ value_growth: vgHist, us_intl: uiHist });
      setMetrics(mets);
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to load data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      await triggerRetrain();
      await loadData();
    } catch (err) {
      setError("Retraining failed: " + (err.message || "Unknown error"));
    } finally {
      setRetraining(false);
    }
  };

  if (loading) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        fontFamily: "system-ui, sans-serif",
        color: "#64748b",
        fontSize: 18,
      }}>
        Loading models and data...
      </div>
    );
  }

  if (error && predictions.length === 0) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        fontFamily: "system-ui, sans-serif",
        color: "#ef4444",
        fontSize: 16,
      }}>
        Error: {error}
      </div>
    );
  }

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, sans-serif", backgroundColor: "#f8fafc", minHeight: "100vh" }}>
      {error && (
        <div style={{ backgroundColor: "#fef2f2", color: "#dc2626", padding: "8px 16px", textAlign: "center", fontSize: 14 }}>
          {error}
        </div>
      )}
      <Dashboard
        predictions={predictions}
        historicalData={historicalData}
        metrics={metrics}
        onRetrain={handleRetrain}
        retraining={retraining}
      />
    </div>
  );
}
