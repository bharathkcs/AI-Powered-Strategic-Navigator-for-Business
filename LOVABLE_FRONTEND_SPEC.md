# 🎨 Lovable AI Frontend Specification

## What to Tell Lovable AI

Copy and paste this prompt into Lovable AI to generate the perfect React/Next.js frontend:

---

## Prompt for Lovable AI:

```
Create a modern, professional web application for "IFB Service Intelligence" - an AI-powered forecasting and analytics platform for service operations.

### Tech Stack:
- Next.js 14 with App Router
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Recharts for data visualization
- Axios for API calls
- React Query for state management

### Design Style:
- Clean, professional corporate design
- Primary color: #2563eb (blue)
- Secondary color: #10b981 (green)
- Accent color: #f59e0b (amber)
- Modern glassmorphism effects
- Responsive design (mobile-first)
- Dark mode toggle

### Pages & Features:

#### 1. Dashboard (Home Page) `/`
- **Header**: Logo, navigation menu, user profile dropdown
- **KPI Cards** (grid layout):
  * Total Service Calls (with trend indicator)
  * Total Revenue (₹ format)
  * Warranty Claim Rate (%)
  * Customer Satisfaction (star rating)
- **Charts Section**:
  * Line chart: Service volume trend (last 6 months)
  * Bar chart: Top 5 performing locations
  * Pie chart: Service type distribution
- **Quick Actions**: Buttons for "Generate Forecast", "View Reports", "Upload Data"

#### 2. Forecasting Page `/forecasting`
- **Forecast Configuration Panel**:
  * Dropdown: Forecast Type (Service Volume, Parts Demand, Revenue, Warranty)
  * Radio buttons: Period (30, 60, 90 days)
  * Dropdown: ML Model (Gradient Boosting, Random Forest)
  * Optional filters: Location, Franchise
  * "Generate Forecast" button (primary, large)
- **Results Display**:
  * Interactive line chart showing historical + forecasted data
  * Model performance metrics (MAE, RMSE, MAPE, R²) in cards
  * AI-generated insights panel (text with markdown support)
  * Download forecast button (CSV export)
- **Forecast History Table**:
  * Columns: Date, Type, Period, Model, Accuracy
  * Actions: View, Download

#### 3. Analytics Page `/analytics`
- **Tabs**:
  * Executive Summary
  * Service Volume Analysis
  * Parts & Inventory
  * Warranty Analysis
- **Executive Summary Tab**:
  * KPI grid (10+ metrics)
  * Trend charts
  * AI insights card
- **Service Volume Tab**:
  * Service type breakdown (pie chart)
  * Duration analysis (histogram)
  * Technician performance table
- **Parts & Inventory Tab**:
  * ABC analysis table with color coding
  * Parts demand heatmap by location
  * Low stock alerts
- **Warranty Tab**:
  * Warranty rate trend line
  * Claims by product category (bar chart)
  * Cost analysis

#### 4. Franchise Performance `/franchise`
- **Performance Tiers Display**:
  * 3 cards: Platinum 🏆, Gold 🥇, Silver 🥈
  * Count of franchises in each tier
- **Franchise Leaderboard Table**:
  * Columns: Rank, Franchise Name, Service Calls, Revenue, Score, Tier
  * Sortable columns
  * Search/filter functionality
- **Individual Franchise View** (modal or detail page):
  * Performance scorecard
  * Comparison vs network average (radar chart)
  * Trend charts (service calls, revenue over time)
  * AI recommendations panel
  * "Download Report" button

#### 5. Inventory Planning `/inventory`
- **ABC Classification View**:
  * 3 sections with tabs: Category A, B, C
  * Parts table with columns: Part Name, Usage, Cost, Category, Stock Level
  * Color-coded rows (A=red, B=yellow, C=green)
- **Location Distribution Heatmap**:
  * Interactive heatmap: Rows=Parts, Columns=Locations, Color=Usage intensity
- **Procurement Recommendations**:
  * Cards showing parts needing reorder
  * Recommended quantities
  * AI procurement insights

#### 6. Reports Page `/reports`
- **Report Type Selector**:
  * Cards for each report type with icons
  * Executive Summary, Operational, Franchise, Inventory, Revenue
- **Report Preview**:
  * Dynamic rendering based on selection
  * Print-friendly view
- **Export Options**:
  * PDF download button
  * CSV download button
  * Email report button

#### 7. Settings Page `/settings`
- **Tabs**: General, API Configuration, Notifications, Users
- **General**: Theme toggle, language, date format
- **API Configuration**: API endpoint URL, API key input
- **Notifications**: Email alerts for forecasts, threshold alerts

### Components to Create:

1. **Navbar**: Sticky top navigation with logo, menu items, search, notifications, profile
2. **Sidebar** (optional): Collapsible left sidebar with icons and labels
3. **MetricCard**: Reusable KPI card with icon, label, value, trend
4. **ChartContainer**: Wrapper for charts with title, subtitle, download button
5. **DataTable**: Advanced table with sorting, filtering, pagination
6. **ForecastChart**: Line chart component for forecast visualization
7. **InsightsPanel**: Card displaying AI-generated insights with markdown
8. **LoadingSpinner**: Custom loading animation
9. **ErrorBoundary**: Error handling component
10. **PerformanceBadge**: Visual tier badge (Platinum/Gold/Silver)

### API Integration:

Base URL: `http://localhost:8000/api/v1`

**Endpoints to connect:**

1. `GET /analytics/summary` → Dashboard KPIs
2. `POST /forecasting/generate` → Generate forecast
3. `GET /forecasting/history` → Forecast history
4. `GET /franchise/performance` → Franchise metrics
5. `GET /inventory/abc-analysis` → Inventory data

**Request/Response Examples:**

```typescript
// Forecast Request
POST /forecasting/generate
{
  "forecast_type": "service_volume",
  "period": 90,
  "model_type": "gradient_boosting",
  "location": null,
  "franchise_id": null
}

// Forecast Response
{
  "forecast_id": 1,
  "forecast_type": "service_volume",
  "period": 90,
  "model_type": "gradient_boosting",
  "forecast_data": [
    {"date": "2024-01-01", "value": 150.5},
    ...
  ],
  "model_metrics": {
    "mae": 10.5,
    "rmse": 15.2,
    "mape": 5.3,
    "r2": 0.85
  },
  "insights": "AI-generated insights text...",
  "created_at": "2024-01-01"
}
```

### Additional Features:

- **Authentication**: Simple API key authentication (header: `X-API-Key`)
- **Error Handling**: Toast notifications for errors
- **Loading States**: Skeletons for loading data
- **Empty States**: Friendly messages when no data
- **Responsive**: Mobile, tablet, desktop breakpoints
- **Accessibility**: ARIA labels, keyboard navigation
- **Animations**: Smooth transitions, fade-ins
- **Export**: CSV/PDF download functionality

### File Upload:

Include a data upload page:
- Drag-and-drop zone for CSV files
- File validation (size, format)
- Upload progress bar
- Preview data table
- Import confirmation

Make it production-ready, beautiful, and highly functional!
```

---

## How to Use This with Lovable AI:

1. **Go to Lovable.dev** or your Lovable AI platform
2. **Start a new project**: Choose "Next.js" or "React"
3. **Paste the entire prompt above** into the chat
4. **Let Lovable AI generate** the initial codebase
5. **Iterate** with follow-up prompts:
   - "Add dark mode toggle"
   - "Make the dashboard more visually appealing"
   - "Add loading skeletons"
   - "Improve mobile responsiveness"

## Post-Generation Steps:

1. **Update API URL** in the generated code:
   ```typescript
   const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
   ```

2. **Install additional dependencies** if needed:
   ```bash
   npm install axios react-query recharts lucide-react
   ```

3. **Connect to your backend**:
   - Update all API calls to point to FastAPI backend
   - Add error handling
   - Test all endpoints

4. **Customize branding**:
   - Replace logo with IFB logo
   - Update color scheme to match IFB brand
   - Add custom fonts if needed

## Example Follow-up Prompts for Lovable AI:

```
"Add a real-time data refresh button on the dashboard"

"Create an advanced filter panel for the analytics page with date range picker"

"Add export to Excel functionality for all tables"

"Implement a notification system for when forecasts are ready"

"Add a comparison view to compare forecasts from different dates"

"Create a mobile-optimized bottom navigation bar"

"Add data caching to reduce API calls"

"Implement optimistic UI updates for better UX"
```

## Tips for Best Results:

1. **Be specific** about component requirements
2. **Provide mockups** or references if you have them
3. **Iterate gradually** - don't ask for everything at once
4. **Test frequently** after each major change
5. **Use shadcn/ui** for consistent, accessible components
6. **Follow Next.js best practices** for performance

---

**Your FastAPI backend is ready to serve this beautiful frontend!** 🚀
