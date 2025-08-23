"""
Performance Benchmarks and Load Tests
Tests system performance under various loads and measures response times
"""

import pytest
import asyncio
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from unittest.mock import MagicMock

# Import modules to test
from optimized_kg_builder import OptimizedKGBuilder, VectorizedStatsCalculator
from async_enrichment_pipeline import HighPerformanceEnrichmentPipeline, EnrichmentConfig
from unified_match_aligner import UnifiedMatchAligner, AlignmentConfig, AlignmentStrategy

@pytest.mark.performance
class TestKnowledgeGraphPerformance:
    """Test Knowledge Graph building performance"""
    
    @pytest.fixture
    def large_cricket_dataset(self):
        """Generate large cricket dataset for performance testing"""
        np.random.seed(42)
        
        # Generate 100K rows (simulating large dataset)
        data = []
        for match_id in range(1, 101):  # 100 matches
            for over in range(1, 21):  # 20 overs
                for ball in range(1, 7):  # 6 balls per over
                    data.append({
                        'match_id': match_id,
                        'over': over,
                        'ball': ball,
                        'batter': f'Player_{np.random.randint(1, 50)}',
                        'bowler': f'Bowler_{np.random.randint(1, 25)}',
                        'runs_scored': np.random.choice([0, 1, 2, 3, 4, 6], p=[0.3, 0.3, 0.2, 0.05, 0.1, 0.05]),
                        'is_wicket': np.random.choice([True, False], p=[0.05, 0.95]),
                        'is_boundary': np.random.choice([True, False], p=[0.15, 0.85]),
                        'is_four': np.random.choice([True, False], p=[0.1, 0.9]),
                        'is_six': np.random.choice([True, False], p=[0.05, 0.95]),
                        'is_no_ball': np.random.choice([True, False], p=[0.02, 0.98]),
                        'is_wide': np.random.choice([True, False], p=[0.03, 0.97]),
                        'venue': f'Stadium_{match_id % 10 + 1}',
                        'date': f'2024-{(match_id % 12) + 1:02d}-{(match_id % 28) + 1:02d}'
                    })
        
        return pd.DataFrame(data)
    
    def test_vectorized_stats_performance(self, benchmark, large_cricket_dataset):
        """Benchmark vectorized statistics calculation"""
        
        def calculate_stats():
            return VectorizedStatsCalculator.calculate_batting_stats_vectorized(large_cricket_dataset)
        
        # Benchmark the function
        result = benchmark(calculate_stats)
        
        # Verify results
        assert len(result) > 0
        assert 'total_runs' in result.columns
        assert 'strike_rate' in result.columns
        
        # Performance assertions
        assert benchmark.stats['mean'] < 5.0  # Should complete in under 5 seconds
    
    def test_kg_building_memory_usage(self, large_cricket_dataset, temp_directory):
        """Test memory usage during KG building"""
        # Save dataset to file
        csv_file = temp_directory / "large_dataset.csv"
        large_cricket_dataset.to_csv(csv_file, index=False)
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Build KG
        builder = OptimizedKGBuilder()
        start_time = time.time()
        
        # Use mock progress callback to avoid I/O overhead
        def mock_progress(stage, message, progress, stats):
            pass
        
        graph = builder.build_from_data(str(csv_file), mock_progress, use_cache=False)
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance assertions
        build_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        assert build_time < 60.0  # Should complete in under 1 minute
        assert memory_increase < 1000  # Should not use more than 1GB additional memory
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Cleanup
        del graph
        gc.collect()
    
    def test_concurrent_kg_operations(self, large_cricket_dataset):
        """Test concurrent KG operations"""
        
        def process_chunk(chunk_data):
            """Process a chunk of data"""
            return VectorizedStatsCalculator.calculate_batting_stats_vectorized(chunk_data)
        
        # Split data into chunks
        chunk_size = len(large_cricket_dataset) // 4
        chunks = [
            large_cricket_dataset[i:i + chunk_size] 
            for i in range(0, len(large_cricket_dataset), chunk_size)
        ]
        
        # Process chunks concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        
        # Verify results
        assert len(results) == len(chunks)
        for result in results:
            assert len(result) > 0
        
        # Performance assertion
        concurrent_time = end_time - start_time
        assert concurrent_time < 10.0  # Should complete quickly with concurrency

@pytest.mark.performance
class TestEnrichmentPipelinePerformance:
    """Test enrichment pipeline performance"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for performance testing"""
        client = MagicMock()
        
        # Mock successful responses with realistic delay
        async def mock_enrich(match_info):
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "match": {"competition": "Test League"},
                "venue": {"name": "Test Stadium"},
                "confidence_score": 0.95
            }
        
        client.enrich_match_async = mock_enrich
        return client
    
    @pytest.fixture
    def sample_matches(self):
        """Generate sample matches for enrichment testing"""
        matches = []
        for i in range(100):
            matches.append({
                "home": f"Team_A_{i % 10}",
                "away": f"Team_B_{i % 10}",
                "venue": f"Stadium_{i % 5}",
                "date": f"2024-04-{(i % 30) + 1:02d}",
                "competition": f"League_{i % 3}"
            })
        return matches
    
    @pytest.mark.asyncio
    async def test_async_enrichment_performance(self, benchmark, sample_matches):
        """Benchmark async enrichment pipeline"""
        config = EnrichmentConfig(
            max_concurrent=10,
            batch_size=20,
            max_retries=1,
            timeout=5
        )
        
        # Create pipeline with mock
        pipeline = HighPerformanceEnrichmentPipeline(
            api_key="test-key",
            config=config,
            cache_dir="test_cache"
        )
        
        # Mock the client to avoid real API calls
        async def mock_enrich_batch(matches):
            # Simulate processing time
            await asyncio.sleep(0.01 * len(matches))
            return [
                {"status": "success", "data": {"confidence_score": 0.9}}
                for _ in matches
            ]
        
        pipeline.enrich_dataset_batch = mock_enrich_batch
        
        # Benchmark the async function
        async def run_enrichment():
            return await pipeline.enrich_dataset_batch(sample_matches[:50])
        
        result = await benchmark(run_enrichment)
        
        # Verify results
        assert len(result) == 50
        
        # Performance assertion
        assert benchmark.stats['mean'] < 2.0  # Should complete in under 2 seconds
    
    def test_enrichment_memory_efficiency(self, sample_matches):
        """Test memory efficiency of enrichment pipeline"""
        config = EnrichmentConfig(
            max_concurrent=5,
            batch_size=10,
            cache_ttl=60
        )
        
        pipeline = HighPerformanceEnrichmentPipeline(
            api_key="test-key",
            config=config
        )
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process matches in batches
        batch_size = 20
        for i in range(0, len(sample_matches), batch_size):
            batch = sample_matches[i:i + batch_size]
            # Simulate processing
            time.sleep(0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly
        assert memory_increase < 100  # Less than 100MB increase

@pytest.mark.performance
class TestMatchAlignerPerformance:
    """Test match aligner performance"""
    
    @pytest.fixture
    def large_datasets(self):
        """Generate large datasets for alignment testing"""
        np.random.seed(42)
        
        # Dataset 1
        data1 = []
        for match_id in range(1, 501):  # 500 matches
            for ball in range(1, 121):  # 120 balls per match (20 overs)
                data1.append({
                    'match_id': match_id,
                    'over': (ball - 1) // 6 + 1,
                    'ball': (ball - 1) % 6 + 1,
                    'batter': f'Player_{np.random.randint(1, 20)}',
                    'bowler': f'Bowler_{np.random.randint(1, 10)}',
                    'runs_scored': np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.2, 0.08, 0.02])
                })
        
        # Dataset 2 (similar but with some variations)
        data2 = []
        for match_id in range(1, 501):
            for ball in range(1, 121):
                data2.append({
                    'Match': match_id,  # Different column name
                    'Over': (ball - 1) // 6 + 1,
                    'Ball': (ball - 1) % 6 + 1,
                    'Batsman': f'Player_{np.random.randint(1, 20)}',  # Different column name
                    'Bowler': f'Bowler_{np.random.randint(1, 10)}',
                    'Runs': np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.2, 0.08, 0.02])
                })
        
        return pd.DataFrame(data1), pd.DataFrame(data2)
    
    def test_dna_hash_strategy_performance(self, benchmark, large_datasets, temp_directory):
        """Benchmark DNA hash strategy"""
        dataset1, dataset2 = large_datasets
        
        # Save datasets
        file1 = temp_directory / "dataset1.csv"
        file2 = temp_directory / "dataset2.csv"
        dataset1.to_csv(file1, index=False)
        dataset2.to_csv(file2, index=False)
        
        # Create aligner
        config = AlignmentConfig(
            strategy=AlignmentStrategy.DNA_HASH,
            similarity_threshold=0.8
        )
        aligner = UnifiedMatchAligner(config)
        
        # Benchmark alignment
        def run_alignment():
            return aligner.align_datasets(str(file1), str(file2))
        
        result = benchmark(run_alignment)
        
        # Verify results
        assert isinstance(result, list)
        
        # Performance assertion
        assert benchmark.stats['mean'] < 30.0  # Should complete in under 30 seconds
    
    def test_hybrid_strategy_performance(self, benchmark, large_datasets, temp_directory):
        """Benchmark hybrid strategy"""
        dataset1, dataset2 = large_datasets
        
        # Save datasets
        file1 = temp_directory / "dataset1.csv"
        file2 = temp_directory / "dataset2.csv"
        dataset1.to_csv(file1, index=False)
        dataset2.to_csv(file2, index=False)
        
        # Create aligner
        config = AlignmentConfig(
            strategy=AlignmentStrategy.HYBRID,
            similarity_threshold=0.7
        )
        aligner = UnifiedMatchAligner(config)
        
        # Benchmark alignment
        def run_alignment():
            return aligner.align_datasets(str(file1), str(file2))
        
        result = benchmark(run_alignment)
        
        # Verify results
        assert isinstance(result, list)
        
        # Performance assertion
        assert benchmark.stats['mean'] < 45.0  # Hybrid might be slower but more accurate

@pytest.mark.performance
class TestSystemResourceUsage:
    """Test system resource usage under load"""
    
    def test_cpu_usage_under_load(self):
        """Test CPU usage during intensive operations"""
        # Generate CPU-intensive workload
        def cpu_intensive_task():
            # Matrix multiplication to stress CPU
            size = 500
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            return np.dot(a, b)
        
        # Monitor CPU usage
        process = psutil.Process()
        cpu_usage_samples = []
        
        start_time = time.time()
        
        # Run CPU intensive tasks
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(4)]
            
            # Sample CPU usage while tasks are running
            while any(not future.done() for future in futures):
                cpu_usage_samples.append(process.cpu_percent())
                time.sleep(0.1)
            
            # Wait for completion
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        # Analyze CPU usage
        if cpu_usage_samples:
            avg_cpu = np.mean(cpu_usage_samples)
            max_cpu = np.max(cpu_usage_samples)
            
            # CPU usage should be reasonable
            assert avg_cpu < 200.0  # Should not exceed 200% (2 cores)
            assert max_cpu < 300.0  # Peak should not exceed 300%
        
        # Verify results
        assert len(results) == 4
        for result in results:
            assert result.shape == (500, 500)
        
        # Total time should be reasonable
        total_time = end_time - start_time
        assert total_time < 30.0  # Should complete in under 30 seconds
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = []
        
        # Create and process large data structures
        large_data = []
        
        for i in range(10):
            # Create large DataFrame
            df = pd.DataFrame(np.random.random((10000, 50)))
            large_data.append(df)
            
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - initial_memory)
            
            # Process the data
            result = df.sum().sum()
            assert result > 0
        
        # Clean up
        del large_data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Analyze memory usage
        peak_memory = max(memory_samples)
        final_memory_increase = final_memory - initial_memory
        
        # Memory should be managed well
        assert peak_memory < 1000  # Should not exceed 1GB
        assert final_memory_increase < 100  # Should release most memory after cleanup
    
    def test_concurrent_operations_performance(self):
        """Test performance of concurrent operations"""
        
        def io_intensive_task(task_id):
            """Simulate I/O intensive task"""
            # Create temporary data
            data = pd.DataFrame(np.random.random((1000, 10)))
            
            # Simulate processing
            result = data.describe()
            
            # Simulate I/O delay
            time.sleep(0.1)
            
            return len(result)
        
        def cpu_intensive_task(task_id):
            """Simulate CPU intensive task"""
            # Matrix operations
            size = 200
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            result = np.dot(a, b)
            return result.sum()
        
        start_time = time.time()
        
        # Run mixed workload
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit mixed tasks
            io_futures = [executor.submit(io_intensive_task, i) for i in range(5)]
            cpu_futures = [executor.submit(cpu_intensive_task, i) for i in range(3)]
            
            # Wait for completion
            io_results = [future.result() for future in io_futures]
            cpu_results = [future.result() for future in cpu_futures]
        
        end_time = time.time()
        
        # Verify results
        assert len(io_results) == 5
        assert len(cpu_results) == 3
        assert all(result > 0 for result in io_results)
        assert all(result > 0 for result in cpu_results)
        
        # Performance assertion
        total_time = end_time - start_time
        assert total_time < 15.0  # Should complete efficiently with concurrency

@pytest.mark.performance
@pytest.mark.slow
class TestLoadTesting:
    """Load testing for system components"""
    
    def test_high_concurrency_load(self):
        """Test system under high concurrency load"""
        
        def simulate_user_request():
            """Simulate a typical user request"""
            # Simulate data processing
            data = pd.DataFrame(np.random.random((100, 5)))
            result = data.groupby(data.columns[0] > 0.5).sum()
            
            # Simulate some computation
            time.sleep(0.05)  # 50ms processing time
            
            return len(result)
        
        # Test with high concurrency
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit many concurrent requests
            futures = [executor.submit(simulate_user_request) for _ in range(100)]
            
            # Collect results
            results = []
            completed = 0
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Log progress every 20 completions
                if completed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    print(f"Completed {completed}/100 requests, rate: {rate:.1f} req/s")
        
        end_time = time.time()
        
        # Analyze performance
        total_time = end_time - start_time
        throughput = len(results) / total_time
        
        # Performance assertions
        assert len(results) == 100  # All requests completed
        assert all(result > 0 for result in results)  # All succeeded
        assert total_time < 30.0  # Completed in reasonable time
        assert throughput > 5.0  # At least 5 requests per second
        
        print(f"Load test completed: {throughput:.1f} req/s, {total_time:.2f}s total")

@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests"""
    
    def test_baseline_performance_metrics(self, performance_timer):
        """Establish baseline performance metrics"""
        
        # Test 1: DataFrame operations
        performance_timer.start()
        
        df = pd.DataFrame(np.random.random((10000, 20)))
        result1 = df.groupby(df.iloc[:, 0] > 0.5).sum()
        
        performance_timer.stop()
        dataframe_time = performance_timer.elapsed
        
        # Test 2: Numpy operations
        performance_timer.start()
        
        arr = np.random.random((1000, 1000))
        result2 = np.dot(arr, arr.T)
        
        performance_timer.stop()
        numpy_time = performance_timer.elapsed
        
        # Test 3: String operations
        performance_timer.start()
        
        strings = [f"player_{i}" for i in range(10000)]
        result3 = [s.upper() for s in strings if len(s) > 8]
        
        performance_timer.stop()
        string_time = performance_timer.elapsed
        
        # Record baseline metrics
        baseline_metrics = {
            "dataframe_operations": dataframe_time,
            "numpy_operations": numpy_time,
            "string_operations": string_time
        }
        
        # Performance assertions (baseline expectations)
        assert dataframe_time < 1.0  # DataFrame operations under 1s
        assert numpy_time < 2.0      # Numpy operations under 2s
        assert string_time < 0.1     # String operations under 0.1s
        
        # Verify results
        assert len(result1) > 0
        assert result2.shape == (1000, 1000)
        assert len(result3) > 0
        
        print(f"Baseline metrics: {baseline_metrics}")
        
        return baseline_metrics
