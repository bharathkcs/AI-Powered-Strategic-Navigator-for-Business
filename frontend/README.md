# IFB Service Intelligence - Frontend

Modern, responsive React frontend for AI-powered forecasting and analytics platform.

## 🚀 Features

- **Data Upload**: CSV file upload with validation and preview
- **Dashboard**: Real-time KPIs and performance metrics
- **AI Forecasting**: 30/60/90-day predictions for service demand, parts, revenue, and warranty
- **Analytics**: Comprehensive insights with location-specific analysis
- **Franchise Performance**: Partner tracking and leaderboard
- **Inventory Management**: ABC analysis and procurement recommendations
- **Revenue Leakage**: Identification and recovery opportunities
- **Reports**: PDF/CSV export functionality

## 🛠️ Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **Charts**: Recharts
- **Routing**: React Router v6
- **File Upload**: React Dropzone
- **CSV Parsing**: PapaParse
- **PDF Generation**: jsPDF

## 📦 Installation

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on http://localhost:8000

### Setup

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Update `.env` with your API URL:
```env
VITE_API_URL=http://localhost:8000/api/v1
```

4. Start development server:
```bash
npm run dev
```

The app will be available at http://localhost:3000

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── Layout.tsx
│   │   └── MetricCard.tsx
│   ├── pages/           # Page components
│   │   ├── Upload.tsx
│   │   ├── Dashboard.tsx
│   │   ├── Forecasting.tsx
│   │   ├── Analytics.tsx
│   │   ├── Franchise.tsx
│   │   ├── Inventory.tsx
│   │   └── Reports.tsx
│   ├── services/        # API integration
│   │   └── api.ts
│   ├── store/          # Global state management
│   │   └── useStore.ts
│   ├── types/          # TypeScript interfaces
│   │   └── index.ts
│   ├── utils/          # Helper functions
│   │   └── helpers.ts
│   ├── App.tsx         # Main app component
│   ├── main.tsx        # Entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML template
├── package.json        # Dependencies
├── tsconfig.json       # TypeScript config
├── vite.config.ts      # Vite config
└── tailwind.config.js  # Tailwind config
```

## 📱 Pages & Features

### 1. Upload Page (`/upload`)
- Drag-and-drop CSV file upload
- File validation and preview
- Progress indicator
- Data import to backend

### 2. Dashboard (`/dashboard`)
- Key performance indicators (KPIs)
- Service volume charts
- Revenue by location
- Warranty analysis
- Quick action buttons

### 3. Forecasting (`/forecasting`)
- Select forecast type (service volume, parts, revenue, warranty)
- Choose period (30/60/90 days)
- ML model selection
- Interactive forecast visualization
- Model performance metrics (MAE, RMSE, MAPE, R²)
- AI-generated insights
- Download forecast data
- Forecast history

### 4. Analytics (`/analytics`)
- Executive summary
- Service type distribution
- Location performance
- Warranty claims analysis
- Revenue leakage identification
- Performance metrics

### 5. Franchise Performance (`/franchise`)
- Performance tier breakdown (Platinum/Gold/Silver/Bronze)
- Franchise leaderboard
- Detailed metrics per franchise
- Network-wide statistics

### 6. Inventory Optimization (`/inventory`)
- ABC category classification
- Parts inventory table
- Usage and cost analysis
- Procurement recommendations
- Location distribution

### 7. Reports (`/reports`)
- Multiple report types
- Date range filtering
- PDF export
- Print functionality
- Report previews

## 🎨 Styling & Theme

The app uses a professional color scheme:
- Primary: Blue (#2563eb)
- Success: Green (#10b981)
- Warning: Yellow (#f59e0b)
- Danger: Red (#ef4444)

Responsive design with mobile-first approach:
- Collapsible sidebar on mobile
- Responsive charts and tables
- Touch-friendly interface

## 🔌 API Integration

The frontend communicates with the FastAPI backend through REST endpoints:

- `POST /api/v1/data/upload` - Upload CSV data
- `POST /api/v1/forecasting/generate` - Generate forecast
- `GET /api/v1/forecasting/history` - Fetch forecast history
- `GET /api/v1/analytics/summary` - Get analytics summary
- `GET /api/v1/franchise/performance` - Fetch franchise data
- `GET /api/v1/inventory/abc-analysis` - Get inventory data
- `GET /api/v1/analytics/revenue-leakages` - Identify leakages

## 📊 Data Flow

1. User uploads CSV file → Backend processes and stores data
2. Frontend fetches data via API calls
3. React Query caches responses
4. Zustand manages global state (upload status, filters)
5. Components render data with charts and tables

## 🚀 Build & Deploy

### Production Build

```bash
npm run build
```

Output will be in `dist/` folder.

### Preview Production Build

```bash
npm run preview
```

### Deploy

Deploy the `dist/` folder to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- Any static hosting service

Don't forget to update `VITE_API_URL` to your production API URL.

## 🧪 Development

### Linting

```bash
npm run lint
```

### Type Checking

TypeScript is configured for strict type checking. The IDE will show type errors in real-time.

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:8000/api/v1` |
| `VITE_API_KEY` | Optional API key | - |

## 🔐 Security Notes

- All API calls go through axios interceptor
- CORS is handled by backend
- No sensitive data stored in localStorage (except optional API key)
- File uploads validated on both client and server

## 🤝 Contributing

This is a production-ready application. To modify:

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit PR with clear description

## 📄 License

Copyright © 2024 IFB Industries. All rights reserved.

## 🆘 Support

For issues or questions:
- Check backend logs for API errors
- Verify backend is running on port 8000
- Ensure CSV file has required columns
- Check browser console for frontend errors

---

**Built with ❤️ for IFB Service Operations**
