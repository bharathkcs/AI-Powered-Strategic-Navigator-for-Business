import { LucideIcon } from 'lucide-react';
import { cn, getTrendColor, getTrendIcon } from '@/utils/helpers';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: number;
  trendLabel?: string;
  subtitle?: string;
  color?: string;
}

const MetricCard = ({
  title,
  value,
  icon: Icon,
  trend,
  trendLabel,
  subtitle,
  color = 'primary',
}: MetricCardProps) => {
  const colorClasses = {
    primary: 'bg-primary-50 text-primary-600',
    success: 'bg-green-50 text-green-600',
    warning: 'bg-yellow-50 text-yellow-600',
    danger: 'bg-red-50 text-red-600',
  };

  return (
    <div className="card hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mb-2">{value}</p>

          {trend !== undefined && (
            <div className="flex items-center space-x-2">
              <span className={cn('text-sm font-medium', getTrendColor(trend))}>
                {getTrendIcon(trend)} {Math.abs(trend).toFixed(1)}%
              </span>
              {trendLabel && <span className="text-xs text-gray-500">{trendLabel}</span>}
            </div>
          )}

          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>

        <div className={cn('p-3 rounded-lg', colorClasses[color as keyof typeof colorClasses] || colorClasses.primary)}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};

export default MetricCard;
