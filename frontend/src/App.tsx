import { useEffect, useState } from "react";
import TextInputCard from "./components/TextInputCard";
import ResultCard from "./components/ResultCard";
import CategoryList from "./components/CategoryList";
import HistoryTable from "./components/HistoryTable";
import { predictNews, fetchCategories, checkHealth } from "./services/api";
import type { PredictResponse } from "./types";

export default function App() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [categories, setCategories] = useState<string[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    checkHealth().then(setApiOnline);
    fetchCategories()
      .then(setCategories)
      .catch(() => setCategories([]));
  }, []);

  const handleClassify = async (text: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictNews(text);
      setResult(res);
      setCategories(res.available_categories);
      setRefreshKey((k) => k + 1);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Nepali News Classifier</h1>
              <p className="text-slate-500 mt-1">
                AI-powered Nepali news category prediction
              </p>
            </div>
            {apiOnline !== null && (
              <span
                className={`text-xs font-medium px-3 py-1.5 rounded-full ${
                  apiOnline
                    ? "bg-emerald-50 text-emerald-700"
                    : "bg-red-50 text-red-700"
                }`}
              >
                {apiOnline ? "API Online" : "API Offline"}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <TextInputCard onClassify={handleClassify} loading={loading} />
        <ResultCard result={result} error={error} />
        <CategoryList categories={categories} activeCategory={result?.category} />
        <HistoryTable refreshKey={refreshKey} />
      </main>

      <footer className="max-w-4xl mx-auto px-6 py-8 text-center text-xs text-slate-400">
        TF-IDF + Logistic Regression · FastAPI · PostgreSQL · React
      </footer>
    </div>
  );
}
