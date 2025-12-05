import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Forecasting from './pages/Forecasting';
import Analytics from './pages/Analytics';
import Franchise from './pages/Franchise';
import Inventory from './pages/Inventory';
import Reports from './pages/Reports';
import { useStore } from './store/useStore';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  const dataLoaded = useStore((state) => state.dataLoaded);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={dataLoaded ? <Dashboard /> : <Navigate to="/upload" replace />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/dashboard" element={dataLoaded ? <Dashboard /> : <Navigate to="/upload" replace />} />
            <Route path="/forecasting" element={dataLoaded ? <Forecasting /> : <Navigate to="/upload" replace />} />
            <Route path="/analytics" element={dataLoaded ? <Analytics /> : <Navigate to="/upload" replace />} />
            <Route path="/franchise" element={dataLoaded ? <Franchise /> : <Navigate to="/upload" replace />} />
            <Route path="/inventory" element={dataLoaded ? <Inventory /> : <Navigate to="/upload" replace />} />
            <Route path="/reports" element={dataLoaded ? <Reports /> : <Navigate to="/upload" replace />} />
          </Routes>
        </Layout>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
