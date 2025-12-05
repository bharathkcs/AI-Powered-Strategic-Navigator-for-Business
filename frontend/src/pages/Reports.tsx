import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { FileText, Download, Calendar, Filter, Printer } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { apiService } from '@/services/api';
import { formatCurrency, formatNumber, formatPercentage, formatDate } from '@/utils/helpers';

type ReportType = 'executive' | 'operational' | 'franchise' | 'inventory' | 'revenue';

const Reports = () => {
  const [selectedReport, setSelectedReport] = useState<ReportType>('executive');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });

  const { data: summary } = useQuery({
    queryKey: ['analytics-summary', dateRange],
    queryFn: () =>
      apiService.getAnalyticsSummary(
        dateRange.start || undefined,
        dateRange.end || undefined
      ),
  });

  const { data: franchises } = useQuery({
    queryKey: ['franchise-performance'],
    queryFn: () => apiService.getFranchisePerformance(),
    enabled: selectedReport === 'franchise',
  });

  const { data: inventory } = useQuery({
    queryKey: ['inventory-abc'],
    queryFn: () => apiService.getABCAnalysis(),
    enabled: selectedReport === 'inventory',
  });

  const { data: leakages } = useQuery({
    queryKey: ['revenue-leakages'],
    queryFn: () => apiService.getRevenueLeakages(),
    enabled: selectedReport === 'revenue',
  });

  const reportTypes = [
    {
      id: 'executive' as ReportType,
      name: 'Executive Summary',
      icon: '📊',
      description: 'High-level overview of key metrics and performance',
    },
    {
      id: 'operational' as ReportType,
      name: 'Operational Report',
      icon: '⚙️',
      description: 'Detailed service operations and efficiency metrics',
    },
    {
      id: 'franchise' as ReportType,
      name: 'Franchise Performance',
      icon: '🏪',
      description: 'Partner performance rankings and comparisons',
    },
    {
      id: 'inventory' as ReportType,
      name: 'Inventory Report',
      icon: '📦',
      description: 'Parts inventory, ABC analysis, and procurement',
    },
    {
      id: 'revenue' as ReportType,
      name: 'Revenue Analysis',
      icon: '💰',
      description: 'Revenue trends, leakages, and optimization',
    },
  ];

  const generatePDF = () => {
    const doc = new jsPDF();
    const title = reportTypes.find((r) => r.id === selectedReport)?.name || 'Report';

    // Title
    doc.setFontSize(20);
    doc.text(title, 14, 22);

    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 30);
    doc.text('IFB Service Intelligence Platform', 14, 36);

    let yPos = 50;

    // Executive Summary
    if (selectedReport === 'executive' && summary) {
      doc.setFontSize(14);
      doc.text('Key Metrics', 14, yPos);
      yPos += 10;

      autoTable(doc, {
        startY: yPos,
        head: [['Metric', 'Value']],
        body: [
          ['Total Service Calls', formatNumber(summary.total_service_calls)],
          ['Total Revenue', formatCurrency(summary.total_revenue)],
          ['Average Service Duration', `${summary.avg_service_duration.toFixed(1)}h`],
          ['Customer Satisfaction', summary.avg_customer_satisfaction.toFixed(1)],
          ['Warranty Claim Rate', formatPercentage(summary.warranty_claim_rate)],
          ['First Call Resolution', formatPercentage(summary.first_call_resolution_rate)],
          ['Revenue per Call', formatCurrency(summary.revenue_per_call)],
          ['Parts Margin', formatPercentage(summary.parts_margin)],
        ],
      });
    }

    // Franchise Report
    if (selectedReport === 'franchise' && franchises) {
      autoTable(doc, {
        startY: yPos,
        head: [['Rank', 'Franchise', 'Region', 'Service Calls', 'Revenue', 'Score', 'Tier']],
        body: franchises.slice(0, 20).map((f) => [
          f.rank.toString(),
          f.franchise_name,
          f.region,
          formatNumber(f.total_service_calls),
          formatCurrency(f.total_revenue),
          f.performance_score.toFixed(1),
          f.tier,
        ]),
      });
    }

    // Inventory Report
    if (selectedReport === 'inventory' && inventory) {
      autoTable(doc, {
        startY: yPos,
        head: [['Part Name', 'Category', 'Usage', 'Cost', 'Revenue', 'Margin']],
        body: inventory.slice(0, 20).map((item) => [
          item.part_name,
          item.category,
          formatNumber(item.total_usage),
          formatCurrency(item.total_cost),
          formatCurrency(item.total_revenue),
          `${item.margin.toFixed(1)}%`,
        ]),
      });
    }

    // Revenue Report
    if (selectedReport === 'revenue' && leakages) {
      autoTable(doc, {
        startY: yPos,
        head: [['Type', 'Description', 'Estimated Amount', 'Services Affected']],
        body: leakages.map((l) => [
          l.type,
          l.description,
          formatCurrency(l.estimated_amount),
          l.count.toString(),
        ]),
      });
    }

    doc.save(`${selectedReport}_report_${Date.now()}.pdf`);
  };

  return (
    <div className="fade-in space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Reports & Downloads</h1>
        <p className="text-gray-600">
          Generate and download comprehensive reports for your operations
        </p>
      </div>

      {/* Date Range Filter */}
      <div className="card">
        <div className="flex items-center mb-4">
          <Calendar className="w-5 h-5 mr-2 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Date Range (Optional)</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="label">Start Date</label>
            <input
              type="date"
              className="input"
              value={dateRange.start}
              onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
            />
          </div>
          <div>
            <label className="label">End Date</label>
            <input
              type="date"
              className="input"
              value={dateRange.end}
              onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={() => setDateRange({ start: '', end: '' })}
              className="btn-secondary w-full"
            >
              <Filter className="w-4 h-4 inline mr-2" />
              Clear Filter
            </button>
          </div>
        </div>
      </div>

      {/* Report Type Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {reportTypes.map((report) => (
          <button
            key={report.id}
            onClick={() => setSelectedReport(report.id)}
            className={`card text-left hover:shadow-md transition-all ${
              selectedReport === report.id
                ? 'border-2 border-primary-500 bg-primary-50'
                : 'border border-gray-200'
            }`}
          >
            <div className="flex items-start">
              <span className="text-4xl mr-4">{report.icon}</span>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">{report.name}</h3>
                <p className="text-sm text-gray-600">{report.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Report Preview */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            {reportTypes.find((r) => r.id === selectedReport)?.name}
          </h3>
          <div className="flex space-x-2">
            <button onClick={generatePDF} className="btn-primary">
              <Download className="w-4 h-4 inline mr-2" />
              Download PDF
            </button>
            <button onClick={() => window.print()} className="btn-secondary">
              <Printer className="w-4 h-4 inline mr-2" />
              Print
            </button>
          </div>
        </div>

        {/* Executive Summary Preview */}
        {selectedReport === 'executive' && summary && (
          <div className="space-y-6">
            <div>
              <h4 className="text-md font-semibold text-gray-900 mb-3">Performance Overview</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-blue-600 font-medium">Service Calls</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {formatNumber(summary.total_service_calls)}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-green-600 font-medium">Revenue</p>
                  <p className="text-2xl font-bold text-green-900">
                    {formatCurrency(summary.total_revenue)}
                  </p>
                </div>
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <p className="text-sm text-yellow-600 font-medium">Satisfaction</p>
                  <p className="text-2xl font-bold text-yellow-900">
                    {summary.avg_customer_satisfaction.toFixed(1)}
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="text-sm text-purple-600 font-medium">FCR Rate</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {formatPercentage(summary.first_call_resolution_rate)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Franchise Report Preview */}
        {selectedReport === 'franchise' && franchises && (
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
                    Revenue
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Tier
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {franchises.slice(0, 10).map((f) => (
                  <tr key={f.franchise_id}>
                    <td className="px-4 py-3 text-sm text-gray-900">{f.rank}</td>
                    <td className="px-4 py-3 text-sm text-gray-900">{f.franchise_name}</td>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {formatCurrency(f.total_revenue)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900">{f.tier}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Other report previews */}
        {selectedReport !== 'executive' && selectedReport !== 'franchise' && (
          <div className="text-center py-12 text-gray-500">
            <FileText className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p>Select a report type and click "Download PDF" to generate the full report</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Reports;
