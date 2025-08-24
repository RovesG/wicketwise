# Purpose: Docker containerization and container orchestration management
# Author: WicketWise AI, Last Modified: 2024

import os
import time
import threading
import subprocess
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import yaml


class ContainerStatus(Enum):
    """Container status types"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"


class DeploymentError(Exception):
    """Deployment related errors"""
    pass


@dataclass
class DockerConfig:
    """Docker configuration settings"""
    image_name: str
    image_tag: str = "latest"
    dockerfile_path: str = "Dockerfile"
    build_context: str = "."
    registry_url: Optional[str] = None
    registry_username: Optional[str] = None
    registry_password: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, str] = field(default_factory=dict)  # container_port: host_port
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    networks: List[str] = field(default_factory=list)
    restart_policy: str = "unless-stopped"
    memory_limit: Optional[str] = None
    cpu_limit: Optional[str] = None
    
    def get_full_image_name(self) -> str:
        """Get full image name with registry and tag"""
        base_name = f"{self.image_name}:{self.image_tag}"
        if self.registry_url:
            return f"{self.registry_url}/{base_name}"
        return base_name


@dataclass
class ContainerMetrics:
    """Container performance metrics"""
    container_id: str
    name: str
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_percent: float
    network_rx_mb: float
    network_tx_mb: float
    block_read_mb: float
    block_write_mb: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'container_id': self.container_id,
            'name': self.name,
            'cpu_percent': self.cpu_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'memory_percent': self.memory_percent,
            'network_rx_mb': self.network_rx_mb,
            'network_tx_mb': self.network_tx_mb,
            'block_read_mb': self.block_read_mb,
            'block_write_mb': self.block_write_mb,
            'timestamp': self.timestamp.isoformat()
        }


class ContainerManager:
    """Docker container management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Container tracking
        self.containers: Dict[str, Dict[str, Any]] = {}
        self.metrics_history: Dict[str, List[ContainerMetrics]] = defaultdict(list)
        
        # Configuration
        self.docker_socket = self.config.get('docker_socket', '/var/run/docker.sock')
        self.metrics_retention_hours = self.config.get('metrics_retention_hours', 24)
        self.monitoring_interval = self.config.get('monitoring_interval_seconds', 30)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Verify Docker availability
        self._verify_docker_available()
    
    def _verify_docker_available(self):
        """Verify Docker is available and accessible"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise DeploymentError("Docker is not available or not accessible")
            
            self.logger.info(f"Docker available: {result.stdout.strip()}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise DeploymentError(f"Docker not found or not accessible: {str(e)}")
    
    def build_image(self, config: DockerConfig, no_cache: bool = False) -> bool:
        """Build Docker image from Dockerfile"""
        try:
            with self.lock:
                build_args = [
                    'docker', 'build',
                    '-t', config.get_full_image_name(),
                    '-f', config.dockerfile_path
                ]
                
                if no_cache:
                    build_args.append('--no-cache')
                
                # Add build context
                build_args.append(config.build_context)
                
                self.logger.info(f"Building Docker image: {config.get_full_image_name()}")
                
                # Execute build
                result = subprocess.run(build_args, capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0:
                    self.logger.info(f"Successfully built image: {config.get_full_image_name()}")
                    return True
                else:
                    self.logger.error(f"Failed to build image: {result.stderr}")
                    raise DeploymentError(f"Image build failed: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            raise DeploymentError("Image build timed out after 30 minutes")
        except Exception as e:
            self.logger.error(f"Error building image: {str(e)}")
            raise DeploymentError(f"Image build error: {str(e)}")
    
    def push_image(self, config: DockerConfig) -> bool:
        """Push Docker image to registry"""
        try:
            with self.lock:
                image_name = config.get_full_image_name()
                
                # Login to registry if credentials provided
                if config.registry_username and config.registry_password:
                    login_result = subprocess.run([
                        'docker', 'login',
                        '-u', config.registry_username,
                        '-p', config.registry_password,
                        config.registry_url or ''
                    ], capture_output=True, text=True, timeout=60)
                    
                    if login_result.returncode != 0:
                        raise DeploymentError(f"Registry login failed: {login_result.stderr}")
                
                # Push image
                self.logger.info(f"Pushing Docker image: {image_name}")
                
                push_result = subprocess.run([
                    'docker', 'push', image_name
                ], capture_output=True, text=True, timeout=1800)
                
                if push_result.returncode == 0:
                    self.logger.info(f"Successfully pushed image: {image_name}")
                    return True
                else:
                    self.logger.error(f"Failed to push image: {push_result.stderr}")
                    raise DeploymentError(f"Image push failed: {push_result.stderr}")
                    
        except subprocess.TimeoutExpired:
            raise DeploymentError("Image push timed out after 30 minutes")
        except Exception as e:
            self.logger.error(f"Error pushing image: {str(e)}")
            raise DeploymentError(f"Image push error: {str(e)}")
    
    def run_container(self, config: DockerConfig, container_name: str, 
                     detached: bool = True) -> str:
        """Run Docker container"""
        try:
            with self.lock:
                run_args = ['docker', 'run']
                
                if detached:
                    run_args.append('-d')
                
                # Container name
                run_args.extend(['--name', container_name])
                
                # Restart policy
                run_args.extend(['--restart', config.restart_policy])
                
                # Environment variables
                for key, value in config.environment_vars.items():
                    run_args.extend(['-e', f"{key}={value}"])
                
                # Port mappings
                for container_port, host_port in config.ports.items():
                    run_args.extend(['-p', f"{host_port}:{container_port}"])
                
                # Volume mounts
                for host_path, container_path in config.volumes.items():
                    run_args.extend(['-v', f"{host_path}:{container_path}"])
                
                # Networks
                for network in config.networks:
                    run_args.extend(['--network', network])
                
                # Resource limits
                if config.memory_limit:
                    run_args.extend(['--memory', config.memory_limit])
                
                if config.cpu_limit:
                    run_args.extend(['--cpus', config.cpu_limit])
                
                # Image name
                run_args.append(config.get_full_image_name())
                
                self.logger.info(f"Running container: {container_name}")
                
                # Execute run
                result = subprocess.run(run_args, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    container_id = result.stdout.strip()
                    
                    # Track container
                    self.containers[container_name] = {
                        'id': container_id,
                        'config': config,
                        'created_at': datetime.now(),
                        'status': ContainerStatus.RUNNING
                    }
                    
                    self.logger.info(f"Successfully started container: {container_name} ({container_id[:12]})")
                    return container_id
                else:
                    self.logger.error(f"Failed to run container: {result.stderr}")
                    raise DeploymentError(f"Container run failed: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            raise DeploymentError("Container run timed out after 5 minutes")
        except Exception as e:
            self.logger.error(f"Error running container: {str(e)}")
            raise DeploymentError(f"Container run error: {str(e)}")
    
    def stop_container(self, container_name: str, timeout: int = 30) -> bool:
        """Stop Docker container"""
        try:
            with self.lock:
                self.logger.info(f"Stopping container: {container_name}")
                
                result = subprocess.run([
                    'docker', 'stop', '-t', str(timeout), container_name
                ], capture_output=True, text=True, timeout=timeout + 10)
                
                if result.returncode == 0:
                    if container_name in self.containers:
                        self.containers[container_name]['status'] = ContainerStatus.EXITED
                    
                    self.logger.info(f"Successfully stopped container: {container_name}")
                    return True
                else:
                    self.logger.error(f"Failed to stop container: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Container stop timed out, force killing: {container_name}")
            return self.kill_container(container_name)
        except Exception as e:
            self.logger.error(f"Error stopping container: {str(e)}")
            return False
    
    def kill_container(self, container_name: str) -> bool:
        """Force kill Docker container"""
        try:
            with self.lock:
                self.logger.info(f"Force killing container: {container_name}")
                
                result = subprocess.run([
                    'docker', 'kill', container_name
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    if container_name in self.containers:
                        self.containers[container_name]['status'] = ContainerStatus.DEAD
                    
                    self.logger.info(f"Successfully killed container: {container_name}")
                    return True
                else:
                    self.logger.error(f"Failed to kill container: {result.stderr}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error killing container: {str(e)}")
            return False
    
    def remove_container(self, container_name: str, force: bool = False) -> bool:
        """Remove Docker container"""
        try:
            with self.lock:
                remove_args = ['docker', 'rm']
                
                if force:
                    remove_args.append('-f')
                
                remove_args.append(container_name)
                
                self.logger.info(f"Removing container: {container_name}")
                
                result = subprocess.run(remove_args, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    if container_name in self.containers:
                        del self.containers[container_name]
                    
                    self.logger.info(f"Successfully removed container: {container_name}")
                    return True
                else:
                    self.logger.error(f"Failed to remove container: {result.stderr}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error removing container: {str(e)}")
            return False
    
    def get_container_status(self, container_name: str) -> Optional[ContainerStatus]:
        """Get container status"""
        try:
            result = subprocess.run([
                'docker', 'inspect', '--format', '{{.State.Status}}', container_name
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                status_str = result.stdout.strip()
                try:
                    return ContainerStatus(status_str)
                except ValueError:
                    self.logger.warning(f"Unknown container status: {status_str}")
                    return None
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting container status: {str(e)}")
            return None
    
    def get_container_logs(self, container_name: str, lines: int = 100, 
                          follow: bool = False) -> str:
        """Get container logs"""
        try:
            log_args = ['docker', 'logs', '--tail', str(lines)]
            
            if follow:
                log_args.append('-f')
            
            log_args.append(container_name)
            
            result = subprocess.run(log_args, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.error(f"Failed to get container logs: {result.stderr}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error getting container logs: {str(e)}")
            return ""
    
    def get_container_metrics(self, container_name: str) -> Optional[ContainerMetrics]:
        """Get container performance metrics"""
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}',
                container_name
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:  # Header + data
                    data_line = lines[1]
                    parts = data_line.split('\t')
                    
                    if len(parts) >= 6:
                        # Parse metrics
                        cpu_percent = float(parts[1].rstrip('%'))
                        
                        # Parse memory usage (format: "used / limit")
                        mem_parts = parts[2].split(' / ')
                        memory_usage_mb = self._parse_memory_value(mem_parts[0])
                        memory_limit_mb = self._parse_memory_value(mem_parts[1])
                        
                        memory_percent = float(parts[3].rstrip('%'))
                        
                        # Parse network I/O (format: "rx / tx")
                        net_parts = parts[4].split(' / ')
                        network_rx_mb = self._parse_memory_value(net_parts[0])
                        network_tx_mb = self._parse_memory_value(net_parts[1])
                        
                        # Parse block I/O (format: "read / write")
                        block_parts = parts[5].split(' / ')
                        block_read_mb = self._parse_memory_value(block_parts[0])
                        block_write_mb = self._parse_memory_value(block_parts[1])
                        
                        return ContainerMetrics(
                            container_id=parts[0],
                            name=container_name,
                            cpu_percent=cpu_percent,
                            memory_usage_mb=memory_usage_mb,
                            memory_limit_mb=memory_limit_mb,
                            memory_percent=memory_percent,
                            network_rx_mb=network_rx_mb,
                            network_tx_mb=network_tx_mb,
                            block_read_mb=block_read_mb,
                            block_write_mb=block_write_mb
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting container metrics: {str(e)}")
            return None
    
    def _parse_memory_value(self, value_str: str) -> float:
        """Parse memory value string to MB"""
        try:
            value_str = value_str.strip()
            
            if value_str.endswith('B'):
                return float(value_str[:-1]) / (1024 * 1024)
            elif value_str.endswith('kB') or value_str.endswith('KB'):
                return float(value_str[:-2]) / 1024
            elif value_str.endswith('MB'):
                return float(value_str[:-2])
            elif value_str.endswith('GB'):
                return float(value_str[:-2]) * 1024
            elif value_str.endswith('TB'):
                return float(value_str[:-2]) * 1024 * 1024
            else:
                # Assume bytes
                return float(value_str) / (1024 * 1024)
                
        except (ValueError, IndexError):
            return 0.0
    
    def list_containers(self, all_containers: bool = False) -> List[Dict[str, Any]]:
        """List Docker containers"""
        try:
            list_args = ['docker', 'ps', '--format', 
                        'table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.CreatedAt}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}']
            
            if all_containers:
                list_args.append('-a')
            
            result = subprocess.run(list_args, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                containers = []
                lines = result.stdout.strip().split('\n')
                
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split('\t')
                        if len(parts) >= 7:
                            containers.append({
                                'id': parts[0],
                                'image': parts[1],
                                'command': parts[2],
                                'created_at': parts[3],
                                'status': parts[4],
                                'ports': parts[5],
                                'name': parts[6]
                            })
                
                return containers
            else:
                self.logger.error(f"Failed to list containers: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error listing containers: {str(e)}")
            return []
    
    def start_monitoring(self):
        """Start container monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Started container monitoring")
    
    def stop_monitoring(self):
        """Stop container monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self.logger.info("Stopped container monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                self._collect_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self):
        """Collect metrics for all tracked containers"""
        with self.lock:
            for container_name in list(self.containers.keys()):
                metrics = self.get_container_metrics(container_name)
                if metrics:
                    self.metrics_history[container_name].append(metrics)
                    
                    # Limit history size
                    max_entries = int((self.metrics_retention_hours * 3600) / self.monitoring_interval)
                    if len(self.metrics_history[container_name]) > max_entries:
                        self.metrics_history[container_name] = self.metrics_history[container_name][-max_entries:]
    
    def get_metrics_history(self, container_name: str, 
                           hours: int = 1) -> List[ContainerMetrics]:
        """Get metrics history for container"""
        with self.lock:
            if container_name not in self.metrics_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [m for m in self.metrics_history[container_name] 
                   if m.timestamp > cutoff_time]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get container manager statistics"""
        with self.lock:
            running_containers = sum(1 for c in self.containers.values() 
                                   if c['status'] == ContainerStatus.RUNNING)
            
            return {
                'total_containers': len(self.containers),
                'running_containers': running_containers,
                'monitoring_active': self._monitoring_active,
                'monitoring_interval': self.monitoring_interval,
                'metrics_retention_hours': self.metrics_retention_hours,
                'containers_with_metrics': len(self.metrics_history)
            }
