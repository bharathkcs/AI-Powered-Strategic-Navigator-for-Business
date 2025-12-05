import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  TrendingUp,
  BarChart3,
  Store,
  Package,
  FileText,
  Menu,
  X
} from 'lucide-react';
import { useState } from 'react';
import { useStore } from '@/store/useStore';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const dataLoaded = useStore((state) => state.dataLoaded);

  const navigation = [
    { name: 'Upload Data', href: '/upload', icon: Upload, enabled: true },
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard, enabled: dataLoaded },
    { name: 'Forecasting', href: '/forecasting', icon: TrendingUp, enabled: dataLoaded },
    { name: 'Analytics', href: '/analytics', icon: BarChart3, enabled: dataLoaded },
    { name: 'Franchise', href: '/franchise', icon: Store, enabled: dataLoaded },
    { name: 'Inventory', href: '/inventory', icon: Package, enabled: dataLoaded },
    { name: 'Reports', href: '/reports', icon: FileText, enabled: dataLoaded },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden mr-2 p-2 rounded-md text-gray-600 hover:bg-gray-100"
              >
                {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-primary-700 rounded-lg flex items-center justify-center text-white font-bold text-lg">
                  IFB
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">IFB Service Intelligence</h1>
                  <p className="text-xs text-gray-500">AI-Powered Forecasting & Analytics</p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {dataLoaded && (
                <div className="hidden sm:flex items-center space-x-2 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span>Data Loaded</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          } lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-40 w-64 bg-white border-r border-gray-200 transition-transform duration-300 ease-in-out`}
        >
          <nav className="mt-5 px-3 space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              const Icon = item.icon;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`
                    flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors
                    ${
                      !item.enabled
                        ? 'text-gray-400 cursor-not-allowed opacity-50'
                        : isActive
                        ? 'bg-primary-50 text-primary-700 border-l-4 border-primary-700'
                        : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                  onClick={(e) => {
                    if (!item.enabled) {
                      e.preventDefault();
                    }
                  }}
                >
                  <Icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main content */}
        <main className="flex-1 lg:ml-0">
          <div className="mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
