import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  UserGroupIcon, 
  TrophyIcon,
  BoltIcon,
  ArrowTrendingUpIcon,
  EyeIcon
} from '@heroicons/react/24/outline';

// Components
import PlayerCard from '../components/PlayerCard';
import MatchCard from '../components/MatchCard';
import StatsCard from '../components/StatsCard';
import RealtimeUpdates from '../components/RealtimeUpdates';
import PersonaSelector from '../components/PersonaSelector';

// Hooks
import { useAuthStore } from '../stores/authStore';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useQuery } from 'react-query';
import { apiClient } from '../services/api';

// Types
interface DashboardStats {
  totalMatches: number;
  totalPlayers: number;
  enrichedMatches: number;
  cacheHitRate: number;
}

interface RecentMatch {
  id: string;
  homeTeam: string;
  awayTeam: string;
  venue: string;
  date: string;
  status: 'live' | 'completed' | 'upcoming';
  winProbability?: number;
}

interface TopPlayer {
  id: string;
  name: string;
  team: string;
  role: string;
  recentForm: number;
  stats: {
    runs?: number;
    wickets?: number;
    average: number;
    strikeRate: number;
  };
}

const Dashboard: React.FC = () => {
  const { user } = useAuthStore();
  const { isConnected, lastMessage } = useWebSocket();
  const [selectedPersona, setSelectedPersona] = useState('analyst');

  // Fetch dashboard data
  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>(
    'dashboard-stats',
    () => apiClient.get('/api/admin/stats').then(res => res.data.stats.data),
    {
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  const { data: recentMatches, isLoading: matchesLoading } = useQuery<RecentMatch[]>(
    'recent-matches',
    () => apiClient.get('/api/matches/recent').then(res => res.data),
    {
      // Mock data for now
      initialData: [
        {
          id: '1',
          homeTeam: 'Mumbai Indians',
          awayTeam: 'Chennai Super Kings',
          venue: 'Wankhede Stadium',
          date: '2024-04-15T19:30:00Z',
          status: 'live' as const,
          winProbability: 65.4
        },
        {
          id: '2',
          homeTeam: 'Royal Challengers Bangalore',
          awayTeam: 'Kolkata Knight Riders',
          venue: 'M Chinnaswamy Stadium',
          date: '2024-04-16T15:30:00Z',
          status: 'upcoming' as const
        }
      ]
    }
  );

  const { data: topPlayers, isLoading: playersLoading } = useQuery<TopPlayer[]>(
    'top-players',
    () => apiClient.get('/api/players/top').then(res => res.data),
    {
      // Mock data for now
      initialData: [
        {
          id: '1',
          name: 'Virat Kohli',
          team: 'RCB',
          role: 'Batsman',
          recentForm: 85,
          stats: {
            runs: 8074,
            average: 37.25,
            strikeRate: 131.97
          }
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
            strikeRate: 19.17
          }
        }
      ]
    }
  );

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 24
      }
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">
              Welcome back, {user?.username}! 🏏
            </h1>
            <p className="text-blue-100">
              Your cricket intelligence dashboard is ready with real-time insights
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* WebSocket Status */}
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
              isConnected ? 'bg-green-500/20 text-green-100' : 'bg-red-500/20 text-red-100'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-400' : 'bg-red-400'
              }`} />
              <span className="text-sm">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
            
            {/* Persona Selector */}
            <PersonaSelector
              selected={selectedPersona}
              onChange={setSelectedPersona}
            />
          </div>
        </div>
      </motion.div>

      {/* Stats Overview */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <motion.div variants={itemVariants}>
          <StatsCard
            title="Total Matches"
            value={stats?.totalMatches || 19653}
            icon={TrophyIcon}
            trend="+12%"
            color="blue"
            loading={statsLoading}
          />
        </motion.div>
        
        <motion.div variants={itemVariants}>
          <StatsCard
            title="Players Analyzed"
            value={stats?.totalPlayers || 2847}
            icon={UserGroupIcon}
            trend="+8%"
            color="green"
            loading={statsLoading}
          />
        </motion.div>
        
        <motion.div variants={itemVariants}>
          <StatsCard
            title="Enriched Data"
            value={stats?.enrichedMatches || 1000}
            icon={BoltIcon}
            trend="+156%"
            color="purple"
            loading={statsLoading}
          />
        </motion.div>
        
        <motion.div variants={itemVariants}>
          <StatsCard
            title="Cache Hit Rate"
            value={`${stats?.cacheHitRate || 85.2}%`}
            icon={ArrowTrendingUpIcon}
            trend="+23%"
            color="orange"
            loading={statsLoading}
          />
        </motion.div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content - 2 columns */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Recent Matches */}
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <TrophyIcon className="w-6 h-6 mr-2 text-blue-600" />
                Recent Matches
              </h2>
              <button className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center">
                View All
                <EyeIcon className="w-4 h-4 ml-1" />
              </button>
            </div>
            
            <div className="space-y-4">
              {matchesLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="animate-pulse">
                      <div className="h-20 bg-gray-200 rounded-lg"></div>
                    </div>
                  ))}
                </div>
              ) : (
                recentMatches?.map((match) => (
                  <MatchCard
                    key={match.id}
                    match={match}
                    persona={selectedPersona}
                  />
                ))
              )}
            </div>
          </motion.div>

          {/* Top Players */}
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <ChartBarIcon className="w-6 h-6 mr-2 text-green-600" />
                Top Performers
              </h2>
              <button className="text-green-600 hover:text-green-700 text-sm font-medium flex items-center">
                View Rankings
                <EyeIcon className="w-4 h-4 ml-1" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {playersLoading ? (
                <div className="col-span-2 space-y-4">
                  {[1, 2].map(i => (
                    <div key={i} className="animate-pulse">
                      <div className="h-24 bg-gray-200 rounded-lg"></div>
                    </div>
                  ))}
                </div>
              ) : (
                topPlayers?.map((player) => (
                  <PlayerCard
                    key={player.id}
                    player={player}
                    persona={selectedPersona}
                    compact
                  />
                ))
              )}
            </div>
          </motion.div>
        </div>

        {/* Sidebar - 1 column */}
        <div className="space-y-8">
          
          {/* Real-time Updates */}
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
              <BoltIcon className="w-6 h-6 mr-2 text-yellow-600" />
              Live Updates
            </h2>
            
            <RealtimeUpdates />
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-6">
              Quick Actions
            </h2>
            
            <div className="space-y-3">
              <button className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
                Analyze New Match
              </button>
              
              <button className="w-full bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors text-sm font-medium">
                Player Comparison
              </button>
              
              <button className="w-full bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium">
                Build Strategy
              </button>
              
              {user?.roles.includes('admin') && (
                <button className="w-full bg-orange-600 text-white px-4 py-2 rounded-lg hover:bg-orange-700 transition-colors text-sm font-medium">
                  Enrich Data
                </button>
              )}
            </div>
          </motion.div>

          {/* System Status */}
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              System Health
            </h2>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">API Gateway</span>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-sm font-medium text-green-600">Healthy</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Knowledge Graph</span>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-sm font-medium text-green-600">Running</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Enrichment Pipeline</span>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></div>
                  <span className="text-sm font-medium text-yellow-600">Processing</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
