import type { ApiError, HistoryItem, PredictResponse } from "../types";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed with status ${res.status}`;
    try {
      const body: ApiError = await res.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail) && body.detail.length > 0) {
        message = body.detail.map((d) => d.msg).join(", ");
      }
    } catch {
      // response body wasn't JSON; keep the generic message
    }
    throw new Error(message);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

export async function predictNews(text: string): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse<PredictResponse>(res);
}

export async function fetchHistory(limit = 50): Promise<HistoryItem[]> {
  const res = await fetch(`${API_URL}/api/history?limit=${limit}`);
  return handleResponse<HistoryItem[]>(res);
}

export async function deleteHistoryItem(id: number): Promise<void> {
  const res = await fetch(`${API_URL}/api/history/${id}`, { method: "DELETE" });
  return handleResponse<void>(res);
}

export async function fetchCategories(): Promise<string[]> {
  const res = await fetch(`${API_URL}/api/categories`);
  const data = await handleResponse<{ categories: string[] }>(res);
  return data.categories;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
