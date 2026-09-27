import { useEffect, useState, useCallback } from "react";
import type { HistoryItem } from "../types";
import { fetchHistory, deleteHistoryItem } from "../services/api";

function truncate(text: string, max = 60) {
  return text.length > max ? text.slice(0, max) + "…" : text;
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function HistoryTable({ refreshKey }: { refreshKey: number }) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHistory();
      setItems(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  const handleDelete = async (id: number) => {
    try {
      await deleteHistoryItem(id);
      setItems((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete item.");
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm font-medium text-slate-700">Classification History</p>
        <button
          onClick={load}
          className="text-xs font-medium text-brand-600 hover:text-brand-700"
        >
          Refresh
        </button>
      </div>

      {loading && <p className="text-sm text-slate-400">Loading history...</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}
      {!loading && !error && items.length === 0 && (
        <p className="text-sm text-slate-400">No predictions yet. Classify some text to get started.</p>
      )}

      {items.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-slate-100">
                <th className="py-2 pr-4 font-medium">Text</th>
                <th className="py-2 pr-4 font-medium">Category</th>
                <th className="py-2 pr-4 font-medium">Confidence</th>
                <th className="py-2 pr-4 font-medium">Date</th>
                <th className="py-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id} className="border-b border-slate-50 last:border-0">
                  <td className="py-2.5 pr-4 text-slate-600">{truncate(item.text)}</td>
                  <td className="py-2.5 pr-4">
                    <span className="text-xs font-medium bg-brand-50 text-brand-700 px-2 py-1 rounded-full">
                      {item.predicted_category}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 text-slate-500">
                    {Math.round(item.confidence * 100)}%
                  </td>
                  <td className="py-2.5 pr-4 text-slate-400 whitespace-nowrap">
                    {formatDate(item.created_at)}
                  </td>
                  <td className="py-2.5">
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="text-xs text-red-500 hover:text-red-700"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
