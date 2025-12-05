import axios, { AxiosInstance } from 'axios';
import type {
  ForecastRequest,
  ForecastResponse,
  AnalyticsSummary,
  FranchisePerformance,
  InventoryItem,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add API key if available
        const apiKey = localStorage.getItem('api_key');
        if (apiKey) {
          config.headers['X-API-Key'] = apiKey;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Upload CSV data
  async uploadData(file: File, onProgress?: (progress: number) => void): Promise<{ message: string; records_count: number }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  // Forecasting endpoints
  async generateForecast(request: ForecastRequest): Promise<ForecastResponse> {
    const response = await this.client.post('/forecasting/generate', request);
    return response.data;
  }

  async getForecastHistory(limit = 10): Promise<ForecastResponse[]> {
    const response = await this.client.get(`/forecasting/history?limit=${limit}`);
    return response.data;
  }

  async getForecastById(id: number): Promise<ForecastResponse> {
    const response = await this.client.get(`/forecasting/${id}`);
    return response.data;
  }

  // Analytics endpoints
  async getAnalyticsSummary(
    startDate?: string,
    endDate?: string,
    location?: string,
    franchiseId?: string
  ): Promise<AnalyticsSummary> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (location) params.append('location', location);
    if (franchiseId) params.append('franchise_id', franchiseId);

    const response = await this.client.get(`/analytics/summary?${params.toString()}`);
    return response.data;
  }

  async getServiceVolumeByType(): Promise<{ service_type: string; count: number }[]> {
    const response = await this.client.get('/analytics/service-volume-by-type');
    return response.data;
  }

  async getRevenueByLocation(): Promise<{ location: string; revenue: number }[]> {
    const response = await this.client.get('/analytics/revenue-by-location');
    return response.data;
  }

  async getWarrantyAnalysis(): Promise<{
    total_claims: number;
    claim_rate: number;
    by_category: { category: string; claims: number }[];
  }> {
    const response = await this.client.get('/analytics/warranty-analysis');
    return response.data;
  }

  // Franchise endpoints
  async getFranchisePerformance(): Promise<FranchisePerformance[]> {
    const response = await this.client.get('/franchise/performance');
    return response.data;
  }

  async getFranchiseById(franchiseId: string): Promise<FranchisePerformance> {
    const response = await this.client.get(`/franchise/${franchiseId}`);
    return response.data;
  }

  // Inventory endpoints
  async getABCAnalysis(): Promise<InventoryItem[]> {
    const response = await this.client.get('/inventory/abc-analysis');
    return response.data;
  }

  async getInventoryByLocation(location: string): Promise<InventoryItem[]> {
    const response = await this.client.get(`/inventory/by-location?location=${location}`);
    return response.data;
  }

  async getProcurementRecommendations(): Promise<{
    part_name: string;
    current_stock: number;
    recommended_stock: number;
    priority: string;
  }[]> {
    const response = await this.client.get('/inventory/procurement-recommendations');
    return response.data;
  }

  // Revenue leakage endpoints
  async getRevenueLeakages(): Promise<{
    type: string;
    description: string;
    estimated_amount: number;
    count: number;
  }[]> {
    const response = await this.client.get('/analytics/revenue-leakages');
    return response.data;
  }

  // Location insights
  async getLocationInsights(): Promise<{
    location: string;
    insights: string;
    metrics: Record<string, number>;
  }[]> {
    const response = await this.client.get('/analytics/location-insights');
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<{ status: string; service: string }> {
    const response = await axios.get('http://localhost:8000/health');
    return response.data;
  }
}

export const apiService = new APIService();
