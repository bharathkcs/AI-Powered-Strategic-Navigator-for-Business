import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Package, AlertCircle, TrendingUp, MapPin } from 'lucide-react';
import { apiService } from '@/services/api';
import { formatCurrency, formatNumber, getCategoryColor } from '@/utils/helpers';
import type { InventoryItem } from '@/types';

const Inventory = () => {
  const [activeCategory, setActiveCategory] = useState<'A' | 'B' | 'C' | 'all'>('all');

  const { data: inventory, isLoading } = useQuery({
    queryKey: ['inventory-abc'],
    queryFn: () => apiService.getABCAnalysis(),
  });

  const { data: recommendations } = useQuery({
    queryKey: ['procurement-recommendations'],
    queryFn: () => apiService.getProcurementRecommendations(),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading inventory data...</p>
        </div>
      </div>
    );
  }

  const filteredInventory =
    activeCategory === 'all'
      ? inventory
      : inventory?.filter((item) => item.category === activeCategory);

  const categoryStats = {
    A: inventory?.filter((item) => item.category === 'A') || [],
    B: inventory?.filter((item) => item.category === 'B') || [],
    C: inventory?.filter((item) => item.category === 'C') || [],
  };

  return (
    <div className="fade-in space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Inventory Optimization</h1>
        <p className="text-gray-600">
          ABC analysis and procurement planning for spare parts management
        </p>
      </div>

      {/* ABC Category Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card bg-gradient-to-br from-red-50 to-red-100 border-red-300">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-red-900">Category A</h3>
            <span className="text-2xl">🔴</span>
          </div>
          <p className="text-3xl font-bold text-red-900">{categoryStats.A.length}</p>
          <p className="text-sm text-red-700 mt-2">High Value - Tight Control</p>
          <p className="text-xs text-red-600 mt-1">
            {formatCurrency(
              categoryStats.A.reduce((sum, item) => sum + item.total_cost, 0)
            )}{' '}
            invested
          </p>
        </div>

        <div className="card bg-gradient-to-br from-yellow-50 to-yellow-100 border-yellow-300">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-yellow-900">Category B</h3>
            <span className="text-2xl">🟡</span>
          </div>
          <p className="text-3xl font-bold text-yellow-900">{categoryStats.B.length}</p>
          <p className="text-sm text-yellow-700 mt-2">Medium Value - Moderate Control</p>
          <p className="text-xs text-yellow-600 mt-1">
            {formatCurrency(
              categoryStats.B.reduce((sum, item) => sum + item.total_cost, 0)
            )}{' '}
            invested
          </p>
        </div>

        <div className="card bg-gradient-to-br from-green-50 to-green-100 border-green-300">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-green-900">Category C</h3>
            <span className="text-2xl">🟢</span>
          </div>
          <p className="text-3xl font-bold text-green-900">{categoryStats.C.length}</p>
          <p className="text-sm text-green-700 mt-2">Low Value - Simple Control</p>
          <p className="text-xs text-green-600 mt-1">
            {formatCurrency(
              categoryStats.C.reduce((sum, item) => sum + item.total_cost, 0)
            )}{' '}
            invested
          </p>
        </div>
      </div>

      {/* Category Filter */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Package className="w-5 h-5 mr-2" />
            Parts Inventory
          </h3>
          <div className="flex space-x-2">
            {(['all', 'A', 'B', 'C'] as const).map((category) => (
              <button
                key={category}
                onClick={() => setActiveCategory(category)}
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                  activeCategory === category
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {category === 'all' ? 'All' : `Category ${category}`}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Part Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Category
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Total Usage
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Total Cost
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Total Revenue
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Margin
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Locations
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Recommended Stock
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredInventory?.map((item, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                    {item.part_name}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`badge ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatNumber(item.total_usage)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatCurrency(item.total_cost)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatCurrency(item.total_revenue)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span
                      className={`font-medium ${
                        item.margin > 30
                          ? 'text-green-600'
                          : item.margin > 15
                          ? 'text-yellow-600'
                          : 'text-red-600'
                      }`}
                    >
                      {item.margin.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {item.locations.length} locations
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatNumber(item.recommended_stock)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Procurement Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-orange-600" />
            Procurement Recommendations
          </h3>

          <div className="space-y-3">
            {recommendations.map((rec, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg border-l-4 ${
                  rec.priority === 'High'
                    ? 'bg-red-50 border-red-500'
                    : rec.priority === 'Medium'
                    ? 'bg-yellow-50 border-yellow-500'
                    : 'bg-blue-50 border-blue-500'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-semibold text-gray-900">{rec.part_name}</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      Current Stock: {rec.current_stock} | Recommended: {rec.recommended_stock}
                    </p>
                  </div>
                  <span
                    className={`badge ${
                      rec.priority === 'High'
                        ? 'bg-red-100 text-red-800 border-red-300'
                        : rec.priority === 'Medium'
                        ? 'bg-yellow-100 text-yellow-800 border-yellow-300'
                        : 'bg-blue-100 text-blue-800 border-blue-300'
                    }`}
                  >
                    {rec.priority} Priority
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Inventory Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <Package className="w-8 h-8 text-primary-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Total Parts</p>
          <p className="text-2xl font-bold text-gray-900">{inventory?.length || 0}</p>
        </div>

        <div className="card">
          <TrendingUp className="w-8 h-8 text-green-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Total Investment</p>
          <p className="text-2xl font-bold text-gray-900">
            {formatCurrency(
              inventory?.reduce((sum, item) => sum + item.total_cost, 0) || 0
            )}
          </p>
        </div>

        <div className="card">
          <MapPin className="w-8 h-8 text-blue-600 mb-3" />
          <p className="text-sm font-medium text-gray-600">Active Locations</p>
          <p className="text-2xl font-bold text-gray-900">
            {new Set(inventory?.flatMap((item) => item.locations)).size || 0}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Inventory;
