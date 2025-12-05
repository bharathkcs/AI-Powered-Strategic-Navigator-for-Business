import { create } from 'zustand';
import type { UploadStatus } from '@/types';

interface AppState {
  // Upload state
  uploadStatus: UploadStatus;
  setUploadStatus: (status: UploadStatus) => void;

  // Data loaded state
  dataLoaded: boolean;
  setDataLoaded: (loaded: boolean) => void;

  // Selected filters
  selectedLocation: string | null;
  selectedFranchise: string | null;
  dateRange: { start: string | null; end: string | null };

  setSelectedLocation: (location: string | null) => void;
  setSelectedFranchise: (franchise: string | null) => void;
  setDateRange: (range: { start: string | null; end: string | null }) => void;

  // Reset filters
  resetFilters: () => void;
}

export const useStore = create<AppState>((set) => ({
  // Upload state
  uploadStatus: {
    uploading: false,
    progress: 0,
  },
  setUploadStatus: (status) => set({ uploadStatus: status }),

  // Data loaded state
  dataLoaded: false,
  setDataLoaded: (loaded) => set({ dataLoaded: loaded }),

  // Selected filters
  selectedLocation: null,
  selectedFranchise: null,
  dateRange: { start: null, end: null },

  setSelectedLocation: (location) => set({ selectedLocation: location }),
  setSelectedFranchise: (franchise) => set({ selectedFranchise: franchise }),
  setDateRange: (range) => set({ dateRange: range }),

  // Reset filters
  resetFilters: () =>
    set({
      selectedLocation: null,
      selectedFranchise: null,
      dateRange: { start: null, end: null },
    }),
}));
