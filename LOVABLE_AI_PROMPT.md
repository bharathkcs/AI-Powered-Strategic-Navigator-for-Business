# 🎨 Lovable AI - IFB Service Intelligence Frontend

## Copy This Entire Prompt to Lovable AI

---

Create a modern, professional web application for **"IFB Service Intelligence"** - an AI-powered forecasting and analytics platform for IFB's service ecosystem.

## Tech Stack Requirements

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui components
- **Charts**: Recharts
- **HTTP Client**: Axios
- **State Management**: React Query (TanStack Query)
- **Form Handling**: React Hook Form with Zod validation
- **Icons**: Lucide React

## Design System

### Colors
- **Primary**: `#2563eb` (Blue 600)
- **Secondary**: `#10b981` (Green 500)
- **Accent**: `#f59e0b` (Amber 500)
- **Danger**: `#ef4444` (Red 500)
- **Success**: `#22c55e` (Green 500)
- **Warning**: `#eab308` (Yellow 500)

### Typography
- **Font Family**: Inter or Geist Sans
- **Headings**: Font weight 600-700
- **Body**: Font weight 400-500

### Design Style
- Clean, professional corporate aesthetic
- Card-based layouts with subtle shadows
- Smooth animations and transitions
- Responsive design (mobile-first approach)
- Light mode (dark mode toggle optional)
- Consistent spacing: 4px, 8px, 12px, 16px, 24px, 32px, 48px

---

## Application Structure

### Navigation
Create a sticky top navbar with:
- **Logo**: "IFB Service Intelligence" with icon
- **Menu Items**: Dashboard, Forecasting, Analytics, Franchise, Inventory, Reports
- **Right Side**: Notification bell, User avatar with dropdown
- **Mobile**: Hamburger menu

### Pages to Create

---

## 1. Dashboard Page - `/`

### Layout
- **Hero Section**: Welcome message + date/time
- **KPI Grid** (4 cards in a row, responsive):

**Card 1: Total Service Calls**
```typescript
{
  title: "Total Service Calls"
  value: number
  icon: Wrench
  trend: { direction: "up" | "down", value: number }
}
```

**Card 2: Total Revenue**
```typescript
{
  title: "Total Revenue"
  value: number (format as ₹ currency)
  icon: DollarSign
  trend: { direction: "up" | "down", value: number }
}
```

**Card 3: Warranty Claim Rate**
```typescript
{
  title: "Warranty Claim Rate"
  value: number (format as %)
  icon: Shield
  trend: { direction: "down" | "up", value: number } // down is good
}
```

**Card 4: Customer Satisfaction**
```typescript
{
  title: "Customer Satisfaction"
  value: number (1-5 scale, show as stars)
  icon: Star
  trend: { direction: "up" | "down", value: number }
}
```

### Charts Section (2 columns on desktop, stack on mobile)

**Chart 1: Service Volume Trend** (Line Chart)
- X-axis: Last 90 days
- Y-axis: Service call count
- Show trend line

**Chart 2: Top Locations** (Bar Chart)
- X-axis: Location names
- Y-axis: Service call count
- Top 5 locations

**Chart 3: Service Type Distribution** (Pie Chart)
- Categories: Installation, Repair, Maintenance, Warranty
- Show percentages

### Quick Actions Section
Three prominent buttons:
1. "Generate Forecast" → Navigate to /forecasting
2. "View Analytics" → Navigate to /analytics
3. "Upload Data" → Open upload modal

### API Integration
```typescript
// GET /api/v1/analytics/summary
interface AnalyticsSummary {
  total_service_calls: number
  total_service_revenue: number
  total_parts_revenue: number
  avg_service_value: number
  warranty_claims: number
  warranty_claim_rate: number
  avg_customer_satisfaction: number
  top_locations: Array<{
    location: string
    count: number
  }>
  insights?: string
}
```

---

## 2. Forecasting Page - `/forecasting`

### Configuration Panel (Left Sidebar or Top Section)

**Form Fields:**
```typescript
interface ForecastRequest {
  forecast_type: "service_volume" | "parts_demand" | "revenue" | "warranty"
  period: 30 | 60 | 90  // days
  model_type: "gradient_boosting" | "random_forest"
  location?: string  // optional
  franchise_id?: string  // optional
}
```

**UI Elements:**
1. **Forecast Type Dropdown**:
   - Service Volume
   - Parts Demand
   - Revenue
   - Warranty Claims

2. **Period Selection** (Radio buttons or tabs):
   - 30 Days
   - 60 Days
   - 90 Days (default)

3. **ML Model Dropdown**:
   - Gradient Boosting (default) - "Recommended for accuracy"
   - Random Forest - "Faster processing"

4. **Optional Filters**:
   - Location (dropdown, nullable)
   - Franchise (dropdown, nullable)

5. **Generate Button** (Large, primary color)
   - Show loading spinner when generating
   - Disable while loading

### Results Display

**Forecast Chart** (Full width, interactive):
- Combined line chart showing:
  - Historical data (solid line, blue)
  - Forecasted data (dashed line, green)
  - Confidence interval (shaded area)
- X-axis: Dates
- Y-axis: Values
- Tooltip on hover

**Model Performance Metrics** (4 cards in a row):
```typescript
interface ModelMetrics {
  mae: number      // Mean Absolute Error
  rmse: number     // Root Mean Squared Error
  mape: number     // Mean Absolute Percentage Error (%)
  r2: number       // R² Score (show as %)
}
```

Display as:
- **MAE**: 10.5 (smaller is better)
- **RMSE**: 15.2 (smaller is better)
- **MAPE**: 5.3% (smaller is better, show as %)
- **R² Score**: 85% (higher is better, show as %)

**AI Insights Panel**:
- Card with markdown-formatted text
- Icon: Lightbulb or Sparkles
- Support markdown formatting from backend

**Action Buttons**:
- "Download Forecast" (CSV export)
- "Save Report" (PDF export)
- "Share" (copy link)

### Forecast History Table

**Columns:**
- Created Date
- Forecast Type
- Period
- Model Used
- Accuracy (R² score)
- Actions (View, Download)

**Features:**
- Sortable columns
- Pagination (10 per page)
- Click row to load that forecast

### API Integration
```typescript
// POST /api/v1/forecasting/generate
interface ForecastResponse {
  forecast_id: number
  forecast_type: string
  period: number
  model_type: string
  forecast_data: Array<{
    date: string  // ISO date
    value: number
    lower_bound?: number
    upper_bound?: number
  }>
  model_metrics: {
    mae: number
    rmse: number
    mape: number
    r2: number
  }
  insights?: string
  created_at: string
}

// GET /api/v1/forecasting/history?limit=10
// Returns: Array<ForecastResponse>
```

---

## 3. Analytics Page - `/analytics`

### Tab Navigation
Create tabs for:
1. Executive Summary (default)
2. Service Volume Analysis
3. Parts & Inventory
4. Warranty Analysis

### Tab 1: Executive Summary

**KPI Grid** (3 rows, 4 columns = 12 metrics):
- Total Service Calls
- Total Service Revenue (₹)
- Total Parts Revenue (₹)
- Average Service Value (₹)
- Warranty Claims Count
- Warranty Claim Rate (%)
- Customer Satisfaction (stars)
- First Call Resolution Rate (%)
- Average Service Duration (hours)
- Total Franchises
- Total Locations
- Total Technicians

**Trend Charts**:
- Monthly revenue trend (line chart)
- Service calls by region (bar chart)

**AI Insights Card**:
- Display insights from API
- Markdown support

### Tab 2: Service Volume Analysis

**Service Type Breakdown** (Donut Chart):
- Installation
- Repair
- Maintenance
- Warranty Service

**Service Duration Distribution** (Histogram):
- X-axis: Duration ranges (0-1h, 1-2h, 2-3h, etc.)
- Y-axis: Count of services

**Top Performing Technicians** (Table):
- Technician ID
- Services Completed
- Avg Satisfaction
- First Call Resolution Rate

### Tab 3: Parts & Inventory

**ABC Analysis Table**:
```typescript
interface InventoryItem {
  part_name: string
  usage_count: number
  total_cost: number
  total_revenue: number
  category: "A" | "B" | "C"
  cumulative_percentage: number
  recommended_stock_level?: number
}
```

**Table Features**:
- Color-coded rows: A=Red, B=Yellow, C=Green
- Sortable columns
- Filter by category

**Parts Demand Heatmap**:
- Rows: Part names
- Columns: Locations
- Color intensity: Usage frequency

### Tab 4: Warranty Analysis

**Warranty Rate Trend** (Line chart):
- X-axis: Months
- Y-axis: Warranty rate %

**Claims by Product Category** (Bar chart):
- X-axis: Product categories
- Y-axis: Number of claims

**Warranty Cost Analysis** (Table):
- Product Category
- Total Claims
- Claim Rate
- Average Cost
- Total Cost Impact

---

## 4. Franchise Performance Page - `/franchise`

### Performance Tiers Summary (3 Cards)

**Card 1: Platinum Tier** 🏆
```typescript
{
  tier: "Platinum"
  count: number
  criteria: "Score > 85"
  color: "#a855f7"  // purple
}
```

**Card 2: Gold Tier** 🥇
```typescript
{
  tier: "Gold"
  count: number
  criteria: "Score 70-85"
  color: "#f59e0b"  // amber
}
```

**Card 3: Silver Tier** 🥈
```typescript
{
  tier: "Silver"
  count: number
  criteria: "Score < 70"
  color: "#94a3b8"  // slate
}
```

### Franchise Leaderboard Table

**Columns:**
- Rank (#)
- Franchise Name
- Location
- Service Calls
- Revenue (₹)
- Warranty Rate (%)
- Performance Score (0-100)
- Tier Badge (colored chip)

**Features:**
- Sortable columns
- Search/filter
- Pagination
- Click row to view details

### Individual Franchise View (Modal or Detail Page)

**Performance Scorecard**:
```typescript
interface FranchisePerformance {
  franchise_id: string
  franchise_name: string
  service_calls: number
  total_revenue: number
  warranty_rate: number
  performance_score: number
  tier: "Platinum" | "Gold" | "Silver"
  rank: number
  insights?: string
}
```

**Display:**
- Score gauge (0-100)
- Tier badge (large)
- Rank indicator

**Comparison Radar Chart**:
- Axes: Service Volume, Revenue, Customer Satisfaction, Efficiency, Quality
- Plot franchise vs network average

**Trend Charts**:
- Service calls over time (line)
- Revenue over time (line)

**AI Recommendations Panel**:
- Show insights from backend
- Action items for improvement

**Action Buttons**:
- "Download Report" (PDF)
- "Contact Franchise"
- "View Details"

### API Integration
```typescript
// GET /api/v1/franchise/performance
interface FranchisePerformanceResponse {
  franchises: Array<{
    franchise_id: string
    franchise_name: string
    service_calls: number
    total_revenue: number
    warranty_rate: number
    performance_score: number
    tier: string
    rank: number
    insights?: string
  }>
}
```

---

## 5. Inventory Planning Page - `/inventory`

### ABC Classification Tabs

**Three Tabs: Category A, Category B, Category C**

**Category A** (Critical - 20% of parts, 80% of value):
- Red theme
- High priority indicator

**Category B** (Important - 30% of parts, 15% of value):
- Yellow/Amber theme
- Medium priority

**Category C** (Regular - 50% of parts, 5% of value):
- Green theme
- Low priority

### Parts Table (for each category)
```typescript
interface InventoryItem {
  part_name: string
  usage_count: number
  total_cost: number
  total_revenue: number
  category: "A" | "B" | "C"
  cumulative_percentage: number
  recommended_stock_level?: number
}
```

**Columns:**
- Part Name
- Usage Count
- Total Cost (₹)
- Total Revenue (₹)
- Cumulative % (of total value)
- Stock Status (badge: In Stock, Low Stock, Out of Stock)
- Recommended Stock Level

### Location Distribution Heatmap

**Interactive Heatmap**:
- Rows: Top 20 part names
- Columns: All locations
- Cell color: Usage intensity (white to dark green)
- Tooltip on hover showing exact count

### Procurement Recommendations Section

**Cards showing:**
```typescript
{
  part_name: string
  current_stock: number
  recommended_stock: number
  urgency: "high" | "medium" | "low"
  estimated_cost: number
}
```

**AI Procurement Insights**:
- Panel showing AI-generated recommendations
- Seasonal trends
- Bulk ordering suggestions

### API Integration
```typescript
// GET /api/v1/inventory/abc-analysis
interface ABCAnalysisResponse {
  items: Array<{
    part_name: string
    usage_count: number
    total_cost: number
    total_revenue: number
    category: "A" | "B" | "C"
    cumulative_percentage: number
    recommended_stock_level?: number
  }>
}
```

---

## 6. Reports Page - `/reports`

### Report Type Selector (Grid of Cards)

**5 Report Type Cards:**
1. **Executive Summary Report**
   - Icon: FileText
   - Description: "Complete overview of operations"

2. **Operational Report**
   - Icon: Activity
   - Description: "Service metrics and performance"

3. **Franchise Report**
   - Icon: Users
   - Description: "Franchise performance analysis"

4. **Inventory Report**
   - Icon: Package
   - Description: "Parts usage and stock levels"

5. **Revenue Report**
   - Icon: DollarSign
   - Description: "Financial analysis and forecasts"

### Report Preview Section

When a report type is selected:
- Show preview in a card
- Dynamic rendering based on selection
- Print-friendly layout

### Export Options (Toolbar)

**Buttons:**
1. **Download PDF** (primary button)
2. **Download CSV** (outline button)
3. **Email Report** (outline button)
4. **Print** (outline button)

### Date Range Selector
- Start Date picker
- End Date picker
- Preset options: Last 7 days, Last 30 days, Last 90 days, Custom

---

## 7. Settings Page - `/settings`

### Tab Navigation
1. General
2. API Configuration
3. Notifications

### General Tab
- **Theme**: Light mode toggle
- **Date Format**: DD/MM/YYYY or MM/DD/YYYY
- **Currency**: ₹ INR (Indian Rupee)
- **Timezone**: Asia/Kolkata

### API Configuration Tab
```typescript
{
  api_endpoint: string  // default: http://localhost:8000/api/v1
  api_key?: string
  timeout: number  // milliseconds
}
```

### Notifications Tab
- Email alerts for:
  - Forecast completion
  - Low stock alerts
  - Warranty rate thresholds
  - Performance issues

---

## 8. Data Upload Page - `/upload`

### File Upload Component

**Drag-and-Drop Zone**:
- Accept: .csv, .xlsx files
- Max size: 50MB
- Visual feedback on drag over

**Upload Process**:
1. File selection
2. File validation (show errors if invalid)
3. Upload progress bar
4. Data preview table (first 10 rows)
5. Column mapping (if needed)
6. Import confirmation button

**Expected CSV Format**:
```csv
service_id,service_date,location,franchise_id,product_category,service_type,service_revenue,parts_revenue,warranty_claim,customer_satisfaction
SVC001,2024-01-15,Mumbai,FR001,Washing Machine,Repair,1500,500,false,4.5
```

**Upload Response**:
```typescript
interface UploadResponse {
  success: boolean
  records_created: number
  records_failed: number
  errors?: string[]
  message: string
}
```

---

## Reusable Components to Create

### 1. MetricCard Component
```tsx
interface MetricCardProps {
  title: string
  value: number | string
  icon: LucideIcon
  trend?: { direction: "up" | "down", value: number }
  format?: "number" | "currency" | "percentage" | "rating"
}
```

### 2. ChartContainer Component
```tsx
interface ChartContainerProps {
  title: string
  subtitle?: string
  children: React.ReactNode
  onDownload?: () => void
}
```

### 3. DataTable Component
```tsx
interface DataTableProps<T> {
  data: T[]
  columns: ColumnDef<T>[]
  searchable?: boolean
  sortable?: boolean
  pagination?: boolean
}
```

### 4. ForecastChart Component
```tsx
interface ForecastChartProps {
  historicalData: { date: string, value: number }[]
  forecastData: { date: string, value: number, lower?: number, upper?: number }[]
  title: string
  yAxisLabel: string
}
```

### 5. InsightsPanel Component
```tsx
interface InsightsPanelProps {
  insights: string  // Markdown formatted
  loading?: boolean
}
```

### 6. LoadingSpinner Component
- Full page loader
- Inline loader
- Button loader

### 7. PerformanceBadge Component
```tsx
interface PerformanceBadgeProps {
  tier: "Platinum" | "Gold" | "Silver"
  size?: "sm" | "md" | "lg"
}
```

### 8. TrendIndicator Component
```tsx
interface TrendIndicatorProps {
  direction: "up" | "down"
  value: number
  format?: "percentage" | "number"
}
```

---

## API Integration Setup

### Base Configuration

Create an API client:
```typescript
// lib/api.ts
import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
})

// Request interceptor (add API key if needed)
apiClient.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('api_key')
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

// Response interceptor (error handling)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle errors globally
    return Promise.reject(error)
  }
)
```

### React Query Setup

```typescript
// lib/queries.ts
import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from './api'

// Get analytics summary
export const useAnalyticsSummary = () => {
  return useQuery({
    queryKey: ['analytics', 'summary'],
    queryFn: async () => {
      const { data } = await apiClient.get('/analytics/summary')
      return data
    }
  })
}

// Generate forecast
export const useGenerateForecast = () => {
  return useMutation({
    mutationFn: async (request: ForecastRequest) => {
      const { data } = await apiClient.post('/forecasting/generate', request)
      return data
    }
  })
}

// Get forecast history
export const useForecastHistory = (limit = 10) => {
  return useQuery({
    queryKey: ['forecasts', 'history', limit],
    queryFn: async () => {
      const { data } = await apiClient.get(`/forecasting/history?limit=${limit}`)
      return data
    }
  })
}
```

---

## Complete API Endpoints Reference

### Analytics
- `GET /api/v1/analytics/summary` → Dashboard KPIs

### Forecasting
- `POST /api/v1/forecasting/generate` → Generate new forecast
- `GET /api/v1/forecasting/history?limit=10` → Get forecast history

### Franchise
- `GET /api/v1/franchise/performance` → Get all franchise metrics

### Inventory
- `GET /api/v1/inventory/abc-analysis` → Get ABC inventory analysis

### System
- `GET /health` → Health check
- `GET /` → API info

---

## Additional Features

### Error Handling
- Toast notifications for all errors
- Specific error messages for:
  - Network errors
  - Validation errors
  - Server errors
  - Timeout errors

### Loading States
- Skeleton loaders for all data-heavy components
- Spinner for button actions
- Progress bars for uploads
- Suspense boundaries for lazy loading

### Empty States
- Friendly illustrations
- Clear call-to-action
- Helpful messages

### Responsive Design
- Mobile: Stack all columns
- Tablet: 2-column layouts
- Desktop: Full layouts
- Breakpoints: 640px, 768px, 1024px, 1280px

### Accessibility
- ARIA labels on all interactive elements
- Keyboard navigation support
- Focus indicators
- Screen reader friendly
- Color contrast compliance (WCAG AA)

### Animations
- Smooth page transitions
- Card hover effects
- Chart animations
- Skeleton loading animations
- Toast slide-ins

### Export Functionality
- CSV export for all tables
- PDF export for reports
- Copy to clipboard for shareable links

---

## Environment Variables

Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_APP_NAME=IFB Service Intelligence
```

---

## File Structure

```
src/
├── app/
│   ├── page.tsx                    # Dashboard
│   ├── forecasting/page.tsx
│   ├── analytics/page.tsx
│   ├── franchise/page.tsx
│   ├── inventory/page.tsx
│   ├── reports/page.tsx
│   ├── settings/page.tsx
│   ├── upload/page.tsx
│   └── layout.tsx
├── components/
│   ├── ui/                         # shadcn components
│   ├── MetricCard.tsx
│   ├── ChartContainer.tsx
│   ├── DataTable.tsx
│   ├── ForecastChart.tsx
│   ├── InsightsPanel.tsx
│   ├── PerformanceBadge.tsx
│   ├── TrendIndicator.tsx
│   ├── Navbar.tsx
│   └── LoadingSpinner.tsx
├── lib/
│   ├── api.ts                      # API client
│   ├── queries.ts                  # React Query hooks
│   ├── utils.ts                    # Utility functions
│   └── types.ts                    # TypeScript types
└── styles/
    └── globals.css
```

---

## Success Criteria

The application should be:
1. ✅ Fully functional with all pages working
2. ✅ Integrated with FastAPI backend
3. ✅ Responsive across all devices
4. ✅ Accessible (WCAG AA compliant)
5. ✅ Production-ready code quality
6. ✅ Beautiful, modern UI
7. ✅ Fast loading with React Query caching
8. ✅ Error-free TypeScript compilation

---

**Make it production-ready, highly functional, and beautiful!** 🚀

