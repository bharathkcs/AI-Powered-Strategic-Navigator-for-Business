import { useQuery } from '@tanstack/react-query';
import { Trophy, Star, TrendingUp, Users, DollarSign } from 'lucide-react';
import { apiService } from '@/services/api';
import {
  formatCurrency,
  formatNumber,
  formatPercentage,
  getPerformanceTierColor,
  getPerformanceTierIcon,
} from '@/utils/helpers';
import type { FranchisePerformance } from '@/types';

const Franchise = () => {
  const { data: franchises, isLoading } = useQuery({
    queryKey: ['franchise-performance'],
    queryFn: () => apiService.getFranchisePerformance(),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading franchise data...</p>
        </div>
      </div>
    );
  }

  const tierCounts = {
    Platinum: franchises?.filter((f) => f.tier === 'Platinum').length || 0,
    Gold: franchises?.filter((f) => f.tier === 'Gold').length || 0,
    Silver: franchises?.filter((f) => f.tier === 'Silver').length || 0,
    Bronze: franchises?.filter((f) => f.tier === 'Bronze').length || 0,
  };

  return (
    <div className="fade-in space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Franchise Performance</h1>
        <p className="text-gray-600">
          Track and compare franchise partner performance across the network
        </p>
      </div>

      {/* Performance Tiers */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card bg-gradient-to-br from-purple-50 to-purple-100 border-purple-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-700">Platinum Tier</p>
              <p className="text-3xl font-bold text-purple-900">{tierCounts.Platinum}</p>
              <p className="text-xs text-purple-600 mt-1">Top performers</p>
            </div>
            <div className="text-4xl">{getPerformanceTierIcon('Platinum')}</div>
          </div>
        </div>

        <div className="card bg-gradient-to-br from-yellow-50 to-yellow-100 border-yellow-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-yellow-700">Gold Tier</p>
              <p className="text-3xl font-bold text-yellow-900">{tierCounts.Gold}</p>
              <p className="text-xs text-yellow-600 mt-1">High performers</p>
            </div>
            <div className="text-4xl">{getPerformanceTierIcon('Gold')}</div>
          </div>
        </div>

        <div className="card bg-gradient-to-br from-gray-50 to-gray-100 border-gray-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-700">Silver Tier</p>
              <p className="text-3xl font-bold text-gray-900">{tierCounts.Silver}</p>
              <p className="text-xs text-gray-600 mt-1">Good performers</p>
            </div>
            <div className="text-4xl">{getPerformanceTierIcon('Silver')}</div>
          </div>
        </div>

        <div className="card bg-gradient-to-br from-orange-50 to-orange-100 border-orange-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-orange-700">Bronze Tier</p>
              <p className="text-3xl font-bold text-orange-900">{tierCounts.Bronze}</p>
              <p className="text-xs text-orange-600 mt-1">Needs improvement</p>
            </div>
            <div className="text-4xl">{getPerformanceTierIcon('Bronze')}</div>
          </div>
        </div>
      </div>

      {/* Franchise Leaderboard */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Trophy className="w-5 h-5 mr-2 text-yellow-600" />
          Franchise Leaderboard
        </h3>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Rank
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Franchise
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Region
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Service Calls
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Revenue
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Satisfaction
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  FCR Rate
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Score
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Tier
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {franchises?.map((franchise) => (
                <tr
                  key={franchise.franchise_id}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <td className="px-4 py-3 text-sm">
                    {franchise.rank <= 3 ? (
                      <span className="text-xl">
                        {franchise.rank === 1
                          ? '🥇'
                          : franchise.rank === 2
                          ? '🥈'
                          : '🥉'}
                      </span>
                    ) : (
                      <span className="text-gray-600 font-medium">{franchise.rank}</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                    {franchise.franchise_name}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">{franchise.region}</td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatNumber(franchise.total_service_calls)}
                  </td>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                    {formatCurrency(franchise.total_revenue)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex items-center">
                      <Star className="w-4 h-4 text-yellow-400 mr-1" />
                      <span className="text-gray-900">
                        {franchise.avg_customer_satisfaction.toFixed(1)}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatPercentage(franchise.first_call_resolution_rate)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className="font-bold text-primary-900">
                      {franchise.performance_score.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`badge ${getPerformanceTierColor(franchise.tier)}`}>
                      {getPerformanceTierIcon(franchise.tier)} {franchise.tier}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Network Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <Users className="w-8 h-8 text-primary-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Total Franchises</p>
          <p className="text-2xl font-bold text-gray-900">{franchises?.length || 0}</p>
        </div>

        <div className="card">
          <DollarSign className="w-8 h-8 text-green-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Network Revenue</p>
          <p className="text-2xl font-bold text-gray-900">
            {formatCurrency(
              franchises?.reduce((sum, f) => sum + f.total_revenue, 0) || 0
            )}
          </p>
        </div>

        <div className="card">
          <TrendingUp className="w-8 h-8 text-blue-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Avg Performance Score</p>
          <p className="text-2xl font-bold text-gray-900">
            {(
              (franchises?.reduce((sum, f) => sum + f.performance_score, 0) || 0) /
              (franchises?.length || 1)
            ).toFixed(1)}
          </p>
        </div>

        <div className="card">
          <Star className="w-8 h-8 text-yellow-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Network Satisfaction</p>
          <p className="text-2xl font-bold text-gray-900">
            {(
              (franchises?.reduce((sum, f) => sum + f.avg_customer_satisfaction, 0) || 0) /
              (franchises?.length || 1)
            ).toFixed(1)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Franchise;
