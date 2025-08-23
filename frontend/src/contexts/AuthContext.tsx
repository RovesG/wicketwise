import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { useAuthStore } from '../stores/authStore';
import { apiClient } from '../services/api';

interface AuthContextType {
  isLoading: boolean;
  validateToken: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const { token, logout, isAuthenticated } = useAuthStore();
  const [isLoading, setIsLoading] = React.useState(true);

  const validateToken = async (): Promise<boolean> => {
    if (!token) {
      setIsLoading(false);
      return false;
    }

    try {
      const response = await apiClient.post('/api/auth/validate');
      setIsLoading(false);
      return response.data.valid;
    } catch (error) {
      console.error('Token validation failed:', error);
      logout();
      setIsLoading(false);
      return false;
    }
  };

  useEffect(() => {
    if (isAuthenticated && token) {
      validateToken();
    } else {
      setIsLoading(false);
    }
  }, []);

  const value = {
    isLoading,
    validateToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
