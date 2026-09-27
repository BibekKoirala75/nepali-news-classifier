import { useState } from "react";
import type { PredictResponse } from "../types";

interface Props {
  result: PredictResponse | null;
  error: string | null;
}

export default function ResultCard({ result, error }: Props) {
  const [copied, setCopied] = useState(false);

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-6">
        <p className="font-medium">Could not classify this text</p>
        <p className="text-sm mt-1">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-dashed border-slate-300 p-6 text-center text-slate-400">
        Paste some Nepali news text and click "Classify News" to see a prediction here.
      </div>
    );
  }

  const confidencePct = Math.round(result.confidence * 100);
  const barColor =
    confidencePct >= 70 ? "bg-emerald-500" : confidencePct >= 40 ? "bg-amber-500" : "bg-red-500";

  const handleCopy = async () => {
    await navigator.clipboard.writeText(
      `Predicted Category: ${result.category}\nConfidence: ${confidencePct}%`
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-500">Predicted Category</p>
          <p className="text-2xl font-bold text-brand-700 mt-1">{result.category}</p>
        </div>
        <button
          onClick={handleCopy}
          className="text-xs font-medium text-slate-500 hover:text-brand-600 border border-slate-200 rounded-md px-2 py-1"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

      <div className="mt-4">
        <div className="flex justify-between text-sm text-slate-500 mb-1">
          <span>Confidence</span>
          <span>{confidencePct}%</span>
        </div>
        <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full ${barColor} transition-all duration-500`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
