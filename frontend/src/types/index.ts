// Core types for IFB Service Intelligence Platform

export interface ServiceData {
  Service_ID: string;
  Service_Date: string;
  Location: string;
  Branch: string;
  Region: string;
  Franchise_ID: string;
  Product_Category: string;
  Service_Type: string;
  Technician_ID: string;
  Customer_ID: string;
  Service_Duration: number;
  Part_Name: string;
  Parts_Used: number;
  Parts_Cost: number;
  Parts_Revenue: number;
  Service_Cost: number;
  Service_Revenue: number;
  Total_Revenue: number;
  Warranty_Claim: number;
  Customer_Satisfaction: number;
  Product_Age_Months: number;
  First_Call_Resolution: number;
  Priority: number;
}

export interface ForecastRequest {
  forecast_type: 'service_volume' | 'parts_demand' | 'revenue' | 'warranty';
  period: 30 | 60 | 90;
  model_type: 'gradient_boosting' | 'random_forest';
  location?: string | null;
  franchise_id?: string | null;
}

export interface ForecastDataPoint {
  date: string;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  mape: number;
  r2: number;
}

export interface ForecastResponse {
  forecast_id: number;
  forecast_type: string;
  period: number;
  model_type: string;
  forecast_data: ForecastDataPoint[];
  model_metrics: ModelMetrics;
  insights: string;
  created_at: string;
  location?: string;
  franchise_id?: string;
}

export interface AnalyticsSummary {
  total_service_calls: number;
  total_revenue: number;
  total_parts_cost: number;
  total_parts_revenue: number;
  avg_service_duration: number;
  avg_customer_satisfaction: number;
  warranty_claim_rate: number;
  first_call_resolution_rate: number;
  revenue_per_call: number;
  parts_margin: number;
}

export interface FranchisePerformance {
  franchise_id: string;
  franchise_name: string;
  tier: 'Platinum' | 'Gold' | 'Silver' | 'Bronze';
  total_service_calls: number;
  total_revenue: number;
  avg_customer_satisfaction: number;
  first_call_resolution_rate: number;
  warranty_claim_rate: number;
  performance_score: number;
  rank: number;
  region: string;
}

export interface InventoryItem {
  part_name: string;
  category: 'A' | 'B' | 'C';
  total_usage: number;
  total_cost: number;
  total_revenue: number;
  margin: number;
  locations: string[];
  recommended_stock: number;
}

export interface RevenueLeakage {
  leakage_type: string;
  description: string;
  estimated_amount: number;
  affected_services: number;
  priority: 'High' | 'Medium' | 'Low';
  recommendation: string;
}

export interface UploadStatus {
  uploading: boolean;
  progress: number;
  fileName?: string;
  error?: string;
  success?: boolean;
  recordsCount?: number;
}

export interface LocationInsight {
  location: string;
  region: string;
  total_calls: number;
  avg_revenue: number;
  top_service_type: string;
  growth_trend: number;
  recommendations: string[];
}

export interface ChartData {
  name: string;
  value: number;
  [key: string]: string | number;
}
