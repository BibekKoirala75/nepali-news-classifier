export interface PredictResponse {
  category: string;
  confidence: number;
  available_categories: string[];
}

export interface HistoryItem {
  id: number;
  text: string;
  predicted_category: string;
  confidence: number;
  created_at: string;
}

export interface ApiError {
  detail: string | { msg: string }[];
}
