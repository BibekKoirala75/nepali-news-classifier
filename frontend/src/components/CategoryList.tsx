interface Props {
  categories: string[];
  activeCategory?: string | null;
}

export default function CategoryList({ categories, activeCategory }: Props) {
  if (categories.length === 0) return null;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <p className="text-sm font-medium text-slate-700 mb-3">Available Categories</p>
      <div className="flex flex-wrap gap-2">
        {categories.map((cat) => (
          <span
            key={cat}
            className={`text-xs font-medium px-3 py-1.5 rounded-full border transition-colors ${
              cat === activeCategory
                ? "bg-brand-600 text-white border-brand-600"
                : "bg-slate-50 text-slate-600 border-slate-200"
            }`}
          >
            {cat}
          </span>
        ))}
      </div>
    </div>
  );
}
