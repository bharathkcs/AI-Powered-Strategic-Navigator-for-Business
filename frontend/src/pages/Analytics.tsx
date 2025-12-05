import { useQuery } from '@tanstack/react-query';
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  MapPin,
  Package,
  DollarSign,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { apiService } from '@/services/api';
import { formatCurrency, formatNumber, formatPercentage } from '@/utils/helpers';
import MetricCard from '@/components/MetricCard';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

const Analytics = () => {
  const { data: summary } = useQuery({
    queryKey: ['analytics-summary'],
    queryFn: () => apiService.getAnalyticsSummary(),
  });

  const { data: serviceVolume } = useQuery({
    queryKey: ['service-volume'],
    queryFn: () => apiService.getServiceVolumeByType(),
  });

  const { data: revenueByLocation } = useQuery({
    queryKey: ['revenue-location'],
    queryFn: () => apiService.getRevenueByLocation(),
  });

  const { data: warrantyAnalysis } = useQuery({
    queryKey: ['warranty-analysis'],
    queryFn: () => apiService.getWarrantyAnalysis(),
  });

  const { data: leakages } = useQuery({
    queryKey: ['revenue-leakages'],
    queryFn: () => apiService.getRevenueLeakages(),
  });

  return (
    <div className="fade-in space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analytics & Insights</h1>
        <p className="text-gray-600">
          Comprehensive analytics across service operations, locations, and performance metrics
        </p>
      </div>

      {/* Executive Summary */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2" />
          Executive Summary
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <p className="text-sm text-blue-600 font-medium">Total Service Calls</p>
            <p className="text-2xl font-bold text-blue-900">
              {formatNumber(summary?.total_service_calls || 0)}
            </p>
            <p className="text-xs text-blue-600 mt-1">Across all locations</p>
          </div>

          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <p className="text-sm text-green-600 font-medium">Total Revenue</p>
            <p className="text-2xl font-bold text-green-900">
              {formatCurrency(summary?.total_revenue || 0)}
            </p>
            <p className="text-xs text-green-600 mt-1">YTD Performance</p>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <p className="text-sm text-purple-600 font-medium">Parts Cost</p>
            <p className="text-2xl font-bold text-purple-900">
              {formatCurrency(summary?.total_parts_cost || 0)}
            </p>
            <p className="text-xs text-purple-600 mt-1">Inventory Investment</p>
          </div>

          <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
            <p className="text-sm text-yellow-600 font-medium">Parts Margin</p>
            <p className="text-2xl font-bold text-yellow-900">
              {formatPercentage(summary?.parts_margin || 0)}
            </p>
            <p className="text-xs text-yellow-600 mt-1">Profitability</p>
          </div>
        </div>
      </div>

      {/* Service Volume Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Service Type Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={serviceVolume || []}
                cx="50%"
                cy="50%"
                labelLine={true}
                label={({ service_type, percent }) =>
                  `${service_type}: ${(percent * 100).toFixed(0)}%`
                }
                outerRadius={100}
                fill="#8884d8"
                dataKey="count"
                nameKey="service_type"
              >
                {(serviceVolume || []).map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Service Volume by Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={serviceVolume || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="service_type" angle={-45} textAnchor="end" height={100} tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Location Performance */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MapPin className="w-5 h-5 mr-2" />
          Revenue by Location
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={(revenueByLocation || []).slice(0, 15)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="location" angle={-45} textAnchor="end" height={120} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(value: number) => formatCurrency(value)} />
            <Bar dataKey="revenue" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Warranty Analysis */}
      {warrantyAnalysis && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2" />
            Warranty Claims Analysis
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <MetricCard
              title="Total Claims"
              value={formatNumber(warrantyAnalysis.total_claims)}
              icon={AlertTriangle}
              color="warning"
            />
            <MetricCard
              title="Claim Rate"
              value={formatPercentage(warrantyAnalysis.claim_rate)}
              icon={TrendingUp}
              color="danger"
            />
            <MetricCard
              title="Categories Affected"
              value={warrantyAnalysis.by_category.length}
              icon={Package}
              color="primary"
            />
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={warrantyAnalysis.by_category}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="claims" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Revenue Leakages */}
      {leakages && leakages.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <DollarSign className="w-5 h-5 mr-2 text-red-600" />
            Revenue Leakage Identification
          </h3>

          <div className="space-y-4">
            {leakages.map((leakage, index) => (
              <div
                key={index}
                className="p-4 border-l-4 border-red-500 bg-red-50 rounded-r-lg"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-semibold text-red-900">{leakage.type}</h4>
                    <p className="text-sm text-red-700 mt-1">{leakage.description}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-red-900">
                      {formatCurrency(leakage.estimated_amount)}
                    </p>
                    <p className="text-xs text-red-600">{leakage.count} services affected</p>
                  </div>
                </div>
              </div>
            ))}

            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm font-medium text-blue-900">
                Total Potential Recovery:{' '}
                <span className="text-xl font-bold">
                  {formatCurrency(leakages.reduce((sum, l) => sum + l.estimated_amount, 0))}
                </span>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Avg Service Duration"
          value={`${(summary?.avg_service_duration || 0).toFixed(1)}h`}
          icon={TrendingUp}
          color="primary"
        />
        <MetricCard
          title="Customer Satisfaction"
          value={(summary?.avg_customer_satisfaction || 0).toFixed(1)}
          icon={TrendingUp}
          subtitle="Out of 5.0"
          color="success"
        />
        <MetricCard
          title="First Call Resolution"
          value={formatPercentage(summary?.first_call_resolution_rate || 0)}
          icon={TrendingUp}
          color="success"
        />
        <MetricCard
          title="Revenue per Call"
          value={formatCurrency(summary?.revenue_per_call || 0)}
          icon={DollarSign}
          color="primary"
        />
      </div>
    </div>
  );
};

export default Analytics;
