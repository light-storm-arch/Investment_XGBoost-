import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export const fetchPredictions = () => api.get("/predictions").then((r) => r.data);
export const fetchHistorical = (comparison) =>
  api.get("/historical", { params: { comparison } }).then((r) => r.data);
export const fetchFeatureImportance = (comparison, horizon) =>
  api.get("/feature-importance", { params: { comparison, horizon } }).then((r) => r.data);
export const fetchModelMetrics = () => api.get("/model-metrics").then((r) => r.data);
export const triggerRetrain = () => api.post("/retrain").then((r) => r.data);
