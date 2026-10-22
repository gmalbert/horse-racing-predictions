/** Shared types matching the precomputed JSON artifacts. */

export interface Manifest {
  generated_at: string;
  stats_windows: number[];
  gb_prediction_dates: string[];
  us_prediction_dates: string[];
  stats_as_of?: string;
}

export interface HorseRow {
  horse: string;
  jockey?: string | null;
  trainer?: string | null;
  win_probability: number;
  win_odds_fractional?: string | null;
  place_probability: number;
  place_odds_fractional?: string | null;
  show_probability: number;
  show_odds_fractional?: string | null;
  top_2_prob?: number;
  top_3_prob?: number;
  win_rank?: number;
  place_rank?: number;
  show_rank?: number;
  age?: number | null;
  weight_lbs?: number | null;
  ofr?: number | string | null;
  form?: string | null;
  draw?: number | null;
  ml_odds?: string | null;
  sire?: string | null;
  dam?: string | null;
  market_odds?: number | null;
  implied_prob?: number | null;
  edge?: number | null;
  ev_per_unit?: number | null;
  best_bookmaker?: string | null;
  odds_trend?: string | null;
  is_value_bet?: boolean | null;
  is_strong_value?: boolean | null;
  betfair_back_odds?: number | null;
  betfair_value_edge?: number | null;
  finish_position?: number | null;
  actual_result?: number | null;
}

export interface RaceDetail {
  course: string | null;
  race_time: string | null;
  race_time_gmt?: string | null;
  race_name?: string | null;
  race_class?: string | null;
  distance_f?: number | null;
  distance_str?: string | null;
  going?: string | null;
  surface?: string | null;
  race_type?: string | null;
  date?: string | null;
  runners: number;
  horses: HorseRow[];
  favorites: {
    win: { horse: string; prob: number };
    place: { horse: string; prob: number };
    show: { horse: string; prob: number };
    favorite_cumulative: {
      horse: string; win: number; top_2: number; top_3: number;
      top_2_increment: number; top_3_increment: number;
    };
  };
  exacta_prob: number | null;
  trifecta_prob: number | null;
  fair_trifecta_odds: number | null;
  predicted_123: string[];
}

export interface DayPayload {
  date: string;
  summary: { horses: number; races: number; courses: number; has_odds: boolean };
  best_bet: (HorseRow & { course?: string; race_time?: string }) | null;
  value_bets: Array<Record<string, unknown>>;
  top25: Array<Record<string, unknown>>;
  races: RaceDetail[];
  filters: {
    classes: string[]; going: string[]; race_types: string[];
    surfaces: string[]; courses: string[];
  };
  model_provenance?: {
    source_model?: string | null;
    model_artifact?: string | null;
    model_feature_count?: number | null;
  };
}

export interface FixturesPayload {
  summary: { total: number; tier1: number; tier2: number; avg_score: number };
  histogram: { bin_edges: number[]; series: Record<string, number[]> };
  top_courses: Array<{ course: string; total_races: number; avg_score: number; max_score: number; tier1_count: number }>;
  fixtures: Array<Record<string, unknown>>;
  courses: string[];
  tiers: string[];
  top_tier1_races: Array<Record<string, unknown>>;
}

export interface SchedulePayload {
  total_upcoming: number;
  courses: number;
  turf_count: number;
  span_days: number;
  fixtures: Array<{ Date: string; Course: string; Time: string; Type: string; Surface: string }>;
}

export interface ModelInsights {
  feature_importance?: Array<{ feature: string; importance: number }>;
  n_features?: number;
  metadata?: Record<string, unknown>;
  calibration?: {
    calibration_date?: string;
    n_calibration_samples?: number;
    metrics?: {
      brier_score_calibrated?: number;
      brier_score_uncalibrated?: number;
      brier_improvement_pct?: number;
    };
  };
  calibrated_model_exists?: boolean;
  diagnostics?: Record<string, unknown>;
}

export interface EntityStatsPayload {
  entity: string;
  window_days: number;
  as_of: string;
  leaderboard: Array<Record<string, unknown>>;
  recent_form_14d: Array<Record<string, unknown>>;
  course_specialists: Array<Record<string, unknown>>;
  partnerships: Array<Record<string, unknown>>;
  going_patterns: Record<string, Array<Record<string, unknown>>>;
  courses: string[];
  form_trends?: Record<string, { dates: string[]; rolling_win_pct: number[]; last30_win_pct: number; last30_wins: number }>;
}

export interface UsTopRaces {
  summary?: {
    races_ranked: number; graded: number; tracks: number;
    avg_top_win_pct: number; avg_margin_pct: number;
  };
  scoring_methodology?: Record<string, unknown>;
  races: Array<Record<string, unknown>>;
}

export interface ExplorerOverview {
  as_of: string;
  summary: Record<string, number>;
  top_horses: Array<Record<string, unknown>>;
  top_jockeys: Array<Record<string, unknown>>;
  top_courses: Array<Record<string, unknown>>;
  distance_distribution: Array<Record<string, unknown>>;
  filter_domains: Record<string, Array<string | number>>;
}

export interface BacktestPayload {
  results?: { generated_at?: string; config?: Record<string, unknown>; folds?: Array<Record<string, unknown>> };
  summary?: {
    n_folds: number; auc_mean: number | null; auc_min: number | null; auc_max: number | null;
    top1_mean: number; top3_mean: number; level_roi_pct_mean: number;
  };
  historical?: {
    n_predictions: number;
    date_range: [string, string] | null;
    segments: Record<string, { accuracy: Array<Record<string, unknown>>; roi: Array<Record<string, unknown>> | null }>;
    calibration: Array<{ predicted_mean: number; actual_rate: number; count: number; bin_label: string }>;
  };
}
