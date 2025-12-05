import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tantml:function_calls>
<invoke name="TrendingUp, Download, Calendar, BarChart3, AlertCircle, CheckCircle } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { apiService } from '@/services/api';
import { formatCurrency, formatNumber, formatDate, downloadCSV } from '@/utils/helpers';
import type { ForecastRequest, ForecastResponse } from '@/types';

const Forecasting = () => {
  const queryClient = useQueryClient();
  const [forecastConfig, setForecastConfig] = useState<ForecastRequest>({
    forecast_type: 'service_volume',
    period: 90,
    model_type: 'gradient_boosting',
    location: null,
    franchise_id: null,
  });

  const [activeForecast, setActiveForecast] = useState<ForecastResponse | null>(null);

  // Fetch forecast history
  const { data: forecastHistory } = useQuery({
    queryKey: ['forecast-history'],
    queryFn: () => apiService.getForecastHistory(),
  });

  // Generate forecast mutation
  const generateForecast = useMutation({
    mutationFn: (config: ForecastRequest) => apiService.generateForecast(config),
    onSuccess: (data) => {
      setActiveForecast(data);
      queryClient.invalidateQueries({ queryKey: ['forecast-history'] });
    },
  });

  const handleGenerateForecast = () => {
    generateForecast.mutate(forecastConfig);
  };

  const handleDownloadForecast = () => {
    if (!activeForecast) return;

    const csvData = activeForecast.forecast_data.map((point) => ({
      Date: point.date,
      Value: point.value,
      Lower_Bound: point.lower_bound || '',
      Upper_Bound: point.upper_bound || '',
    }));

    downloadCSV(
      csvData,
      `forecast_${activeForecast.forecast_type}_${activeForecast.period}days_${Date.now()}.csv`
    );
  };

  const getForecastLabel = (type: string): string => {
    const labels = {
      service_volume: 'Service Volume',
      parts_demand: 'Parts Demand',
      revenue: 'Revenue',
      warranty: 'Warranty Claims',
    };
    return labels[type as keyof typeof labels] || type;
  };

  return (
    <div className="fade-in space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">AI-Powered Forecasting</h1>
        <p className="text-gray-600">
          Generate 30/60/90-day forecasts for service demand, parts usage, and revenue
        </p>
      </div>

      {/* Forecast Configuration */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2" />
          Configure Forecast
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Forecast Type */}
          <div>
            <label className="label">Forecast Type</label>
            <select
              className="input"
              value={forecastConfig.forecast_type}
              onChange={(e) =>
                setForecastConfig({
                  ...forecastConfig,
                  forecast_type: e.target.value as ForecastRequest['forecast_type'],
                })
              }
            >
              <option value="service_volume">Service Volume</option>
              <option value="parts_demand">Parts Demand</option>
              <option value="revenue">Revenue</option>
              <option value="warranty">Warranty Claims</option>
            </select>
          </div>

          {/* Period */}
          <div>
            <label className="label">Forecast Period</label>
            <select
              className="input"
              value={forecastConfig.period}
              onChange={(e) =>
                setForecastConfig({
                  ...forecastConfig,
                  period: parseInt(e.target.value) as ForecastRequest['period'],
                })
              }
            >
              <option value={30}>30 Days</option>
              <option value={60}>60 Days</option>
              <option value={90}>90 Days</option>
            </select>
          </div>

          {/* Model Type */}
          <div>
            <label className="label">ML Model</label>
            <select
              className="input"
              value={forecastConfig.model_type}
              onChange={(e) =>
                setForecastConfig({
                  ...forecastConfig,
                  model_type: e.target.value as ForecastRequest['model_type'],
                })
              }
            >
              <option value="gradient_boosting">Gradient Boosting</option>
              <option value="random_forest">Random Forest</option>
            </select>
          </div>

          {/* Generate Button */}
          <div className="flex items-end">
            <button
              onClick={handleGenerateForecast}
              disabled={generateForecast.isPending}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {generateForecast.isPending ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white inline-block mr-2"></div>
                  Generating...
                </>
              ) : (
                <>
                  <TrendingUp className="inline w-4 h-4 mr-2" />
                  Generate Forecast
                </>
              )}
            </button>
          </div>
        </div>

        {generateForecast.isError && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
            <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">
              Failed to generate forecast. Please ensure data is uploaded and try again.
            </p>
          </div>
        )}
      </div>

      {/* Forecast Results */}
      {activeForecast && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {getForecastLabel(activeForecast.forecast_type)} - {activeForecast.period} Days
              </h3>
              <p className="text-sm text-gray-600">
                Model: {activeForecast.model_type.replace('_', ' ').toUpperCase()} | Generated:{' '}
                {formatDate(activeForecast.created_at)}
              </p>
            </div>
            <button onClick={handleDownloadForecast} className="btn-secondary">
              <Download className="w-4 h-4 mr-2 inline" />
              Download CSV
            </button>
          </div>

          {/* Model Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <p className="text-xs text-blue-600 font-medium">MAE</p>
              <p className="text-xl font-bold text-blue-900">
                {activeForecast.model_metrics.mae.toFixed(2)}
              </p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
              <p className="text-xs text-green-600 font-medium">RMSE</p>
              <p className="text-xl font-bold text-green-900">
                {activeForecast.model_metrics.rmse.toFixed(2)}
              </p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
              <p className="text-xs text-yellow-600 font-medium">MAPE</p>
              <p className="text-xl font-bold text-yellow-900">
                {activeForecast.model_metrics.mape.toFixed(2)}%
              </p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
              <p className="text-xs text-purple-600 font-medium">R² Score</p>
              <p className="text-xl font-bold text-purple-900">
                {activeForecast.model_metrics.r2.toFixed(3)}
              </p>
            </div>
          </div>

          {/* Forecast Chart */}
          <div className="mb-6">
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={activeForecast.forecast_data}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) => new Date(date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}
                  tick={{ fontSize: 12 }}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value: number) =>
                    activeForecast.forecast_type === 'revenue'
                      ? formatCurrency(value)
                      : formatNumber(value)
                  }
                  labelFormatter={(date) => formatDate(date)}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#2563eb"
                  fillOpacity={1}
                  fill="url(#colorValue)"
                  name="Forecast"
                />
                {activeForecast.forecast_data[0]?.lower_bound && (
                  <>
                    <Area
                      type="monotone"
                      dataKey="lower_bound"
                      stroke="#94a3b8"
                      strokeDasharray="5 5"
                      fill="none"
                      name="Lower Bound"
                    />
                    <Area
                      type="monotone"
                      dataKey="upper_bound"
                      stroke="#94a3b8"
                      strokeDasharray="5 5"
                      fill="none"
                      name="Upper Bound"
                    />
                  </>
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* AI Insights */}
          {activeForecast.insights && (
            <div className="bg-gradient-to-r from-primary-50 to-blue-50 p-6 rounded-lg border border-primary-200">
              <div className="flex items-start">
                <div className="bg-primary-600 p-2 rounded-lg mr-4">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-primary-900 mb-2">AI-Generated Insights</h4>
                  <p className="text-sm text-gray-700 whitespace-pre-line">{activeForecast.insights}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Forecast History */}
      {forecastHistory && forecastHistory.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Calendar className="w-5 h-5 mr-2" />
            Recent Forecasts
          </h3>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Date
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Period
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Model
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Accuracy (R²)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {forecastHistory.map((forecast) => (
                  <tr key={forecast.forecast_id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {formatDate(forecast.created_at)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {getForecastLabel(forecast.forecast_type)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">{forecast.period} days</td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {forecast.model_type.replace('_', ' ')}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={`badge ${
                          forecast.model_metrics.r2 >= 0.8
                            ? 'bg-green-100 text-green-800 border-green-300'
                            : forecast.model_metrics.r2 >= 0.6
                            ? 'bg-yellow-100 text-yellow-800 border-yellow-300'
                            : 'bg-red-100 text-red-800 border-red-300'
                        }`}
                      >
                        {forecast.model_metrics.r2.toFixed(3)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <button
                        onClick={() => setActiveForecast(forecast)}
                        className="text-primary-600 hover:text-primary-800 font-medium"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Forecasting;
