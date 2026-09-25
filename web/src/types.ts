// Miroir de api/sheet.py (unités SI).

export interface Metric {
  value: number | null;
  confidence: number;
  manually_corrected: boolean;
  to_check: boolean;
}

export interface Split {
  distance: number;
  time: Metric;
  segment_time: number | null;
  segment_speed: number | null;
}

export interface Section {
  label: string;
  stroke_rate: Metric;
  stroke_length: Metric;
  stroke_index: Metric;
}

export interface StrokeReading {
  stroke_rate_change_pct: number | null;
  stroke_length_change_pct: number | null;
  stroke_index_change_pct: number | null;
  diagnosis: string | null;
}

export interface ProfilePoint {
  d: number;
  v: number | null;
  to_check: boolean;
}

export interface RaceSheet {
  is_demo: boolean;
  title: string;
  reaction_time: Metric;
  splits: Split[];
  underwater_start: Metric;
  underwater_turn: Metric;
  sections: Section[];
  stroke_reading: StrokeReading;
  turn_time: Metric;
  finish_speed: Metric;
  velocity_profile: ProfilePoint[];
  benchmarks_available: boolean;
}
