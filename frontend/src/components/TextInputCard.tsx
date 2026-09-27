import { useState } from "react";

interface Props {
  onClassify: (text: string) => void;
  loading: boolean;
}

const MAX_CHARS = 20000;

export default function TextInputCard({ onClassify, loading }: Props) {
  const [text, setText] = useState("");

  const handleSubmit = () => {
    if (text.trim().length === 0) return;
    onClassify(text);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <label htmlFor="news-text" className="block text-sm font-medium text-slate-700 mb-2">
        Nepali News Text
      </label>
      <textarea
        id="news-text"
        value={text}
        onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
        placeholder="Paste Nepali news article or headline..."
        rows={8}
        className="w-full resize-y rounded-lg border border-slate-300 p-3 text-base leading-relaxed focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
      />
      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-slate-400">
          {text.length} / {MAX_CHARS} characters
        </span>
        <button
          onClick={handleSubmit}
          disabled={loading || text.trim().length === 0}
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <>
              <span className="h-4 w-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
              Classifying...
            </>
          ) : (
            "Classify News"
          )}
        </button>
      </div>
    </div>
  );
}
