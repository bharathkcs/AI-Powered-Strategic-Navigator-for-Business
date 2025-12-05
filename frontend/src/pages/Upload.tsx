import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload as UploadIcon, FileText, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import Papa from 'papaparse';
import { apiService } from '@/services/api';
import { useStore } from '@/store/useStore';
import type { ServiceData } from '@/types';

const Upload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ServiceData[]>([]);
  const [error, setError] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const setDataLoaded = useStore((state) => state.setDataLoaded);

  const validateCSV = (data: any[]): boolean => {
    if (data.length === 0) {
      setError('CSV file is empty');
      return false;
    }

    const requiredColumns = [
      'Service_ID',
      'Service_Date',
      'Location',
      'Product_Category',
      'Service_Type',
      'Total_Revenue',
    ];

    const headers = Object.keys(data[0]);
    const missingColumns = requiredColumns.filter((col) => !headers.includes(col));

    if (missingColumns.length > 0) {
      setError(`Missing required columns: ${missingColumns.join(', ')}`);
      return false;
    }

    return true;
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const uploadedFile = acceptedFiles[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setError('');
    setUploadSuccess(false);

    // Parse CSV for preview
    Papa.parse(uploadedFile, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (validateCSV(results.data)) {
          setPreview(results.data.slice(0, 10) as ServiceData[]);
        }
      },
      error: (error) => {
        setError(`Error parsing CSV: ${error.message}`);
      },
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
    },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError('');
    setUploadProgress(0);

    try {
      const result = await apiService.uploadData(file, (progress) => {
        setUploadProgress(progress);
      });

      setUploadSuccess(true);
      setDataLoaded(true);

      setTimeout(() => {
        window.location.href = '/dashboard';
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to upload file. Please try again.');
      setUploadSuccess(false);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Service Data</h1>
        <p className="text-gray-600">
          Upload your IFB service data CSV file to start analyzing and forecasting
        </p>
      </div>

      {/* Upload Area */}
      <div className="card mb-6">
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
            ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
            ${file ? 'bg-green-50 border-green-300' : ''}
          `}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center">
            {file ? (
              <>
                <CheckCircle className="w-16 h-16 text-green-600 mb-4" />
                <p className="text-lg font-medium text-gray-900 mb-2">{file.name}</p>
                <p className="text-sm text-gray-600">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </>
            ) : (
              <>
                <UploadIcon className="w-16 h-16 text-gray-400 mb-4" />
                <p className="text-lg font-medium text-gray-900 mb-2">
                  {isDragActive ? 'Drop the file here' : 'Drag & drop your CSV file here'}
                </p>
                <p className="text-sm text-gray-600 mb-4">or click to browse</p>
                <p className="text-xs text-gray-500">Supported format: CSV (Max 50MB)</p>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
            <XCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {uploadSuccess && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-start">
            <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-green-700">
              File uploaded successfully! Redirecting to dashboard...
            </p>
          </div>
        )}

        {uploading && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {file && !uploading && !uploadSuccess && (
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleUpload}
              className="btn-primary px-6 py-3"
            >
              Upload & Process Data
            </button>
          </div>
        )}
      </div>

      {/* Data Preview */}
      {preview.length > 0 && (
        <div className="card">
          <div className="flex items-center mb-4">
            <FileText className="w-5 h-5 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Data Preview</h3>
            <span className="ml-2 text-sm text-gray-500">(First 10 rows)</span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Service ID
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Date
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Location
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Product
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Service Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Revenue
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {preview.map((row, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm text-gray-900">{row.Service_ID}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">{row.Service_Date}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">{row.Location}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">{row.Product_Category}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">{row.Service_Type}</td>
                    <td className="px-4 py-3 text-sm text-gray-900 font-medium">
                      ₹{parseFloat(row.Total_Revenue.toString()).toLocaleString('en-IN')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-start">
        <AlertCircle className="w-5 h-5 text-blue-600 mr-3 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-700">
          <p className="font-medium mb-1">Expected CSV Format:</p>
          <p>
            Your CSV should contain columns like Service_ID, Service_Date, Location, Branch, Region,
            Franchise_ID, Product_Category, Service_Type, Parts_Cost, Service_Revenue, Total_Revenue,
            Warranty_Claim, Customer_Satisfaction, etc.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Upload;
