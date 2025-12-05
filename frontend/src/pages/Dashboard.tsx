import { useQuery } from '@tanstack/react-query';
import {
  DollarSign,
  TrendingUp,
  Users,
  Package,
  AlertTriangle,
  Star,
  Activity,
  CheckCircle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
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
import MetricCard from '@/components/MetricCard';
import { apiService } from '@/services/api';
import { formatCurrency, formatNumber, formatPercentage } from '@/utils/helpers';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

const Dashboard = () => {
  // Fetch analytics summary
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['analytics-summary'],
    queryFn: () => apiService.getAnalyticsSummary(),
  });

  // Fetch service volume by type
  const { data: serviceVolume, isLoading: volumeLoading } = useQuery({
    queryKey: ['service-volume'],
    queryFn: () => apiService.getServiceVolumeByType(),
  });

  // Fetch revenue by location
  const { data: revenueByLocation, isLoading: revenueLoading } = useQuery({
    queryKey: ['revenue-location'],
    queryFn: () => apiService.getRevenueByLocation(),
  });

  // Fetch warranty analysis
  const { data: warrantyData, isLoading: warrantyLoading } = useQuery({
    queryKey: ['warranty-analysis'],
    queryFn: () => apiService.getWarrantyAnalysis(),
  });

  if (summaryLoading || volumeLoading || revenueLoading || warrantyLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fade-in space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
        <p className="text-gray-600">Overview of IFB service operations and key metrics</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Service Calls"
          value={formatNumber(summary?.total_service_calls || 0)}
          icon={Activity}
          color="primary"
          trend={8.5}
          trendLabel="vs last month"
        />
        <MetricCard
          title="Total Revenue"
          value={formatCurrency(summary?.total_revenue || 0)}
          icon={DollarSign}
          color="success"
          trend={12.3}
          trendLabel="vs last month"
        />
        <MetricCard
          title="Warranty Claim Rate"
          value={formatPercentage(summary?.warranty_claim_rate || 0)}
          icon={AlertTriangle}
          color="warning"
          trend={-2.1}
          trendLabel="vs last month"
        />
        <MetricCard
          title="Customer Satisfaction"
          value={(summary?.avg_customer_satisfaction || 0).toFixed(1)}
          icon={Star}
          color="primary"
          subtitle="Out of 5.0"
          trend={3.5}
          trendLabel="vs last month"
        />
      </div>

      {/* Secondary KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Avg Service Duration"
          value={`${(summary?.avg_service_duration || 0).toFixed(1)}h`}
          icon={TrendingUp}
          color="primary"
        />
        <MetricCard
          title="First Call Resolution"
          value={formatPercentage(summary?.first_call_resolution_rate || 0)}
          icon={CheckCircle}
          color="success"
        />
        <MetricCard
          title="Revenue per Call"
          value={formatCurrency(summary?.revenue_per_call || 0)}
          icon={DollarSign}
          color="primary"
        />
        <MetricCard
          title="Parts Margin"
          value={formatPercentage(summary?.parts_margin || 0)}
          icon={Package}
          color="success"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Service Volume by Type */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Service Volume by Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={serviceVolume || []}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ service_type, percent }) =>
                  `${service_type}: ${(percent * 100).toFixed(0)}%`
                }
                outerRadius={80}
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

        {/* Top Performing Locations */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Top Performing Locations by Revenue
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={(revenueByLocation || []).slice(0, 8)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="location"
                angle={-45}
                textAnchor="end"
                height={100}
                tick={{ fontSize: 12 }}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(value: number) => formatCurrency(value)}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="revenue" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Warranty Analysis */}
      {warrantyData && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Warranty Claims by Category</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
              <p className="text-sm text-red-600 font-medium">Total Claims</p>
              <p className="text-2xl font-bold text-red-900">{formatNumber(warrantyData.total_claims)}</p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
              <p className="text-sm text-yellow-600 font-medium">Claim Rate</p>
              <p className="text-2xl font-bold text-yellow-900">
                {formatPercentage(warrantyData.claim_rate)}
              </p>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-600 font-medium">Categories Affected</p>
              <p className="text-2xl font-bold text-blue-900">{warrantyData.by_category.length}</p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={warrantyData.by_category}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Bar dataKey="claims" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <a
            href="/forecasting"
            className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-center"
          >
            <TrendingUp className="w-8 h-8 mx-auto mb-2 text-primary-600" />
            <p className="font-medium text-gray-900">Generate Forecast</p>
          </a>
          <a
            href="/analytics"
            className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-center"
          >
            <Activity className="w-8 h-8 mx-auto mb-2 text-primary-600" />
            <p className="font-medium text-gray-900">View Analytics</p>
          </a>
          <a
            href="/franchise"
            className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-center"
          >
            <Users className="w-8 h-8 mx-auto mb-2 text-primary-600" />
            <p className="font-medium text-gray-900">Franchise Performance</p>
          </a>
          <a
            href="/inventory"
            className="p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-center"
          >
            <Package className="w-8 h-8 mx-auto mb-2 text-primary-600" />
            <p className="font-medium text-gray-900">Inventory Planning</p>
          </a>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
