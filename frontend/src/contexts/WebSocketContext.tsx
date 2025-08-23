import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuthStore } from '../stores/authStore';
import toast from 'react-hot-toast';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  lastMessage: any;
  sendMessage: (event: string, data: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const { user, token, isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (isAuthenticated && user && token) {
      // Connect to WebSocket
      const newSocket = io(`ws://127.0.0.1:5005/api/ws/${user.id}`, {
        auth: {
          token: token,
        },
        transports: ['websocket'],
      });

      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        toast.success('Connected to live updates');
      });

      newSocket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        toast.error('Disconnected from live updates');
      });

      newSocket.on('connection_established', (data) => {
        console.log('Connection established:', data);
        setLastMessage(data);
      });

      newSocket.on('enrichment_started', (data) => {
        console.log('Enrichment started:', data);
        setLastMessage(data);
        toast.success(`Match enrichment started: ${data.matches_count} matches`);
      });

      newSocket.on('kg_build_started', (data) => {
        console.log('KG build started:', data);
        setLastMessage(data);
        toast.success('Knowledge graph build started');
      });

      newSocket.on('echo', (data) => {
        console.log('Echo received:', data);
        setLastMessage(data);
      });

      newSocket.on('error', (error) => {
        console.error('WebSocket error:', error);
        toast.error('WebSocket connection error');
      });

      setSocket(newSocket);

      return () => {
        newSocket.close();
      };
    } else {
      // Cleanup if not authenticated
      if (socket) {
        socket.close();
        setSocket(null);
        setIsConnected(false);
      }
    }
  }, [isAuthenticated, user, token]);

  const sendMessage = (event: string, data: any) => {
    if (socket && isConnected) {
      socket.emit(event, data);
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  };

  const value = {
    socket,
    isConnected,
    lastMessage,
    sendMessage,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};
