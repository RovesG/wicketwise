import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { useAuthStore } from '../stores/authStore';
import toast from 'react-hot-toast';

// Create axios instance
export const apiClient: AxiosInstance = axios.create({
  baseURL: 'http://127.0.0.1:5005',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config: AxiosRequestConfig) => {
    const token = useAuthStore.getState().token;
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    const { response } = error;

    if (response?.status === 401) {
      // Unauthorized - clear auth and redirect to login
      useAuthStore.getState().logout();
      toast.error('Session expired. Please login again.');
      window.location.href = '/login';
    } else if (response?.status === 403) {
      // Forbidden
      toast.error('You do not have permission to perform this action.');
    } else if (response?.status === 429) {
      // Rate limited
      toast.error('Too many requests. Please try again later.');
    } else if (response?.status >= 500) {
      // Server error
      toast.error('Server error. Please try again later.');
    } else if (error.code === 'NETWORK_ERROR' || !response) {
      // Network error
      toast.error('Network error. Please check your connection.');
    }

    return Promise.reject(error);
  }
);

// API service functions
export const authAPI = {
  login: async (username: string, password: string) => {
    const response = await apiClient.post('/api/auth/login', {
      username,
      password,
    });
    return response.data;
  },

  validateToken: async () => {
    const response = await apiClient.post('/api/auth/validate');
    return response.data;
  },
};

export const adminAPI = {
  getServicesStatus: async () => {
    const response = await apiClient.get('/api/admin/services');
    return response.data;
  },

  enrichMatches: async (data: {
    max_matches: number;
    priority_competitions: string[];
    force_refresh: boolean;
  }) => {
    const response = await apiClient.post('/api/admin/enrich-matches', data);
    return response.data;
  },

  buildKnowledgeGraph: async (dataPath: string) => {
    const response = await apiClient.post('/api/kg/build', null, {
      params: { data_path: dataPath },
    });
    return response.data;
  },
};

export const playerAPI = {
  queryPlayer: async (data: {
    player_name: string;
    format?: string;
    venue?: string;
  }) => {
    const response = await apiClient.post('/api/query/player', data);
    return response.data;
  },

  getTopPlayers: async () => {
    // Mock data for now - would be real API call
    return {
      data: [
        {
          id: '1',
          name: 'Virat Kohli',
          team: 'RCB',
          role: 'Batsman',
          recentForm: 85,
          stats: {
            runs: 8074,
            average: 37.25,
            strikeRate: 131.97,
          },
        },
        {
          id: '2',
          name: 'Jasprit Bumrah',
          team: 'MI',
          role: 'Bowler',
          recentForm: 92,
          stats: {
            wickets: 159,
            average: 24.43,
            strikeRate: 19.17,
          },
        },
      ],
    };
  },
};

export const matchAPI = {
  getRecentMatches: async () => {
    // Mock data for now - would be real API call
    return {
      data: [
        {
          id: '1',
          homeTeam: 'Mumbai Indians',
          awayTeam: 'Chennai Super Kings',
          venue: 'Wankhede Stadium',
          date: '2024-04-15T19:30:00Z',
          status: 'live' as const,
          winProbability: 65.4,
        },
        {
          id: '2',
          homeTeam: 'Royal Challengers Bangalore',
          awayTeam: 'Kolkata Knight Riders',
          venue: 'M Chinnaswamy Stadium',
          date: '2024-04-16T15:30:00Z',
          status: 'upcoming' as const,
        },
      ],
    };
  },

  alignDatasets: async (data: {
    dataset1_path: string;
    dataset2_path: string;
    strategy: string;
    similarity_threshold: number;
  }) => {
    const response = await apiClient.post('/api/alignment/align', data);
    return response.data;
  },
};

export const systemAPI = {
  getHealth: async () => {
    const response = await apiClient.get('/api/health');
    return response.data;
  },
};
