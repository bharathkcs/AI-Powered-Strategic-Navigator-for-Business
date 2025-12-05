// Utility helper functions

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(amount);
};

export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('en-IN').format(num);
};

export const formatPercentage = (value: number, decimals = 1): string => {
  return `${value.toFixed(decimals)}%`;
};

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(date);
};

export const getTrendIcon = (trend: number): string => {
  if (trend > 0) return '↑';
  if (trend < 0) return '↓';
  return '→';
};

export const getTrendColor = (trend: number, inverse = false): string => {
  const positive = inverse ? trend < 0 : trend > 0;
  if (positive) return 'text-green-600';
  if (trend < 0 && !inverse) return 'text-red-600';
  if (trend > 0 && inverse) return 'text-red-600';
  return 'text-gray-600';
};

export const getPerformanceTierColor = (tier: string): string => {
  switch (tier) {
    case 'Platinum':
      return 'bg-purple-100 text-purple-800 border-purple-300';
    case 'Gold':
      return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    case 'Silver':
      return 'bg-gray-100 text-gray-800 border-gray-300';
    case 'Bronze':
      return 'bg-orange-100 text-orange-800 border-orange-300';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

export const getPerformanceTierIcon = (tier: string): string => {
  switch (tier) {
    case 'Platinum':
      return '💎';
    case 'Gold':
      return '🥇';
    case 'Silver':
      return '🥈';
    case 'Bronze':
      return '🥉';
    default:
      return '📊';
  }
};

export const getCategoryColor = (category: 'A' | 'B' | 'C'): string => {
  switch (category) {
    case 'A':
      return 'bg-red-100 text-red-800 border-red-300';
    case 'B':
      return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    case 'C':
      return 'bg-green-100 text-green-800 border-green-300';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

export const getPriorityColor = (priority: string): string => {
  switch (priority.toLowerCase()) {
    case 'high':
      return 'bg-red-100 text-red-800 border-red-300';
    case 'medium':
      return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    case 'low':
      return 'bg-green-100 text-green-800 border-green-300';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-300';
  }
};

export const cn = (...classes: (string | undefined | null | false)[]): string => {
  return classes.filter(Boolean).join(' ');
};

export const downloadCSV = (data: any[], filename: string): void => {
  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map((row) =>
      headers.map((header) => JSON.stringify(row[header] ?? '')).join(',')
    ),
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const calculateGrowthRate = (current: number, previous: number): number => {
  if (previous === 0) return 0;
  return ((current - previous) / previous) * 100;
};

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};
