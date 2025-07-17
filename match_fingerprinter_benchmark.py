# Purpose: Performance benchmark system for match finger printer validation
# Author: Assistant, Last Modified: 2024

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from hybrid_match_aligner import HybridMatchAligner
import time
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchFingerprinterBenchmark:
    """Comprehensive benchmark system for match finger printer validation."""
    
    def __init__(self, nvplay_path: str, decimal_path: str):
        """Initialize benchmark system."""
        self.nvplay_path = nvplay_path
        self.decimal_path = decimal_path
        self.results = {}
        self.aligner = None
        
    def initialize_aligner(self):
        """Initialize the match aligner."""
        print("🔧 Initializing Match Aligner...")
        start_time = time.time()
        
        self.aligner = HybridMatchAligner(self.nvplay_path, self.decimal_path, None)
        
        # Force fallback configuration for consistent testing
        self.aligner.openai_api_key = None
        config = self.aligner.generate_llm_configuration()
        config.similarity_threshold = 0.4  # Optimized threshold
        config.fingerprint_length = 15
        self.aligner.config = config
        
        init_time = time.time() - start_time
        
        print(f"✅ Aligner initialized in {init_time:.2f}s")
        print(f"📊 NVPlay records: {len(self.aligner.nvplay_df):,}")
        print(f"📊 Decimal records: {len(self.aligner.decimal_df):,}")
        
        return init_time
        
    def benchmark_fingerprint_extraction(self, sample_sizes: List[int] = [100, 500, 1000, 2000]):
        """Benchmark fingerprint extraction performance."""
        print("\n📊 BENCHMARKING FINGERPRINT EXTRACTION")
        print("=" * 50)
        
        extraction_results = []
        
        for sample_size in sample_sizes:
            print(f"\n🔍 Testing sample size: {sample_size}")
            
            # Create sample data
            nvplay_sample = self.aligner.nvplay_df.head(sample_size)
            decimal_sample = self.aligner.decimal_df.head(sample_size)
            
            # Backup original data
            original_nvplay = self.aligner.nvplay_df
            original_decimal = self.aligner.decimal_df
            
            # Set sample data
            self.aligner.nvplay_df = nvplay_sample
            self.aligner.decimal_df = decimal_sample
            
            # Benchmark extraction
            start_time = time.time()
            nvplay_fps, decimal_fps = self.aligner.extract_fingerprints()
            extraction_time = time.time() - start_time
            
            # Calculate metrics
            nvplay_fingerprints = len(nvplay_fps)
            decimal_fingerprints = len(decimal_fps)
            records_per_second = sample_size / extraction_time if extraction_time > 0 else 0
            
            result = {
                'sample_size': sample_size,
                'nvplay_fingerprints': nvplay_fingerprints,
                'decimal_fingerprints': decimal_fingerprints,
                'extraction_time': extraction_time,
                'records_per_second': records_per_second
            }
            
            extraction_results.append(result)
            
            print(f"  ⏱️  Extraction time: {extraction_time:.2f}s")
            print(f"  📈 Records/second: {records_per_second:.0f}")
            print(f"  🎯 NVPlay fingerprints: {nvplay_fingerprints}")
            print(f"  🎯 Decimal fingerprints: {decimal_fingerprints}")
            
            # Restore original data
            self.aligner.nvplay_df = original_nvplay
            self.aligner.decimal_df = original_decimal
            
        self.results['fingerprint_extraction'] = extraction_results
        return extraction_results
        
    def benchmark_similarity_calculation(self, num_comparisons: int = 1000):
        """Benchmark similarity calculation performance."""
        print(f"\n🧮 BENCHMARKING SIMILARITY CALCULATION")
        print("=" * 50)
        
        # Extract sample fingerprints
        sample_data = self.aligner.nvplay_df.head(500)
        self.aligner.nvplay_df = sample_data
        self.aligner.decimal_df = self.aligner.decimal_df.head(500)
        
        nvplay_fps, decimal_fps = self.aligner.extract_fingerprints()
        
        if not nvplay_fps or not decimal_fps:
            print("❌ No fingerprints available for similarity testing")
            return
            
        # Get sample fingerprints
        nvplay_list = list(nvplay_fps.values())
        decimal_list = list(decimal_fps.values())
        
        # Benchmark similarity calculations
        start_time = time.time()
        similarities = []
        
        for i in range(min(num_comparisons, len(nvplay_list) * len(decimal_list))):
            nvplay_idx = i % len(nvplay_list)
            decimal_idx = (i // len(nvplay_list)) % len(decimal_list)
            
            similarity = self.aligner.calculate_fuzzy_similarity(
                nvplay_list[nvplay_idx], 
                decimal_list[decimal_idx]
            )
            similarities.append(similarity)
            
        calculation_time = time.time() - start_time
        
        # Calculate metrics
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        comparisons_per_second = num_comparisons / calculation_time if calculation_time > 0 else 0
        
        result = {
            'num_comparisons': num_comparisons,
            'calculation_time': calculation_time,
            'comparisons_per_second': comparisons_per_second,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity
        }
        
        print(f"  ⏱️  Calculation time: {calculation_time:.2f}s")
        print(f"  📈 Comparisons/second: {comparisons_per_second:.0f}")
        print(f"  📊 Average similarity: {avg_similarity:.3f}")
        print(f"  📊 Std deviation: {std_similarity:.3f}")
        print(f"  📊 Min similarity: {min_similarity:.3f}")
        print(f"  📊 Max similarity: {max_similarity:.3f}")
        
        self.results['similarity_calculation'] = result
        return result
        
    def benchmark_match_finding(self, sample_sizes: List[int] = [100, 500, 1000]):
        """Benchmark match finding with different thresholds."""
        print(f"\n🎯 BENCHMARKING MATCH FINDING")
        print("=" * 50)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        match_results = []
        
        for sample_size in sample_sizes:
            print(f"\n🔍 Testing sample size: {sample_size}")
            
            # Create sample data
            nvplay_sample = self.aligner.nvplay_df.head(sample_size)
            decimal_sample = self.aligner.decimal_df.head(sample_size)
            
            # Backup and set sample data
            original_nvplay = self.aligner.nvplay_df
            original_decimal = self.aligner.decimal_df
            
            self.aligner.nvplay_df = nvplay_sample
            self.aligner.decimal_df = decimal_sample
            
            for threshold in thresholds:
                self.aligner.config.similarity_threshold = threshold
                
                start_time = time.time()
                matches = self.aligner.find_matches(use_llm_config=False)
                match_time = time.time() - start_time
                
                # Calculate statistics
                if matches:
                    similarities = [m['similarity_score'] for m in matches]
                    avg_similarity = np.mean(similarities)
                    high_confidence = sum(1 for s in similarities if s > 0.8)
                    medium_confidence = sum(1 for s in similarities if 0.6 <= s <= 0.8)
                    low_confidence = sum(1 for s in similarities if s < 0.6)
                else:
                    avg_similarity = 0
                    high_confidence = medium_confidence = low_confidence = 0
                
                result = {
                    'sample_size': sample_size,
                    'threshold': threshold,
                    'num_matches': len(matches),
                    'match_time': match_time,
                    'avg_similarity': avg_similarity,
                    'high_confidence': high_confidence,
                    'medium_confidence': medium_confidence,
                    'low_confidence': low_confidence
                }
                
                match_results.append(result)
                
                print(f"  Threshold {threshold}: {len(matches)} matches in {match_time:.2f}s (avg: {avg_similarity:.3f})")
            
            # Restore original data
            self.aligner.nvplay_df = original_nvplay
            self.aligner.decimal_df = original_decimal
            
        self.results['match_finding'] = match_results
        return match_results
        
    def benchmark_name_matching(self, test_cases: List[Tuple[str, str]] = None):
        """Benchmark enhanced name matching performance."""
        print(f"\n👥 BENCHMARKING NAME MATCHING")
        print("=" * 50)
        
        if test_cases is None:
            test_cases = [
                ('JR Philippe', 'Josh Philippe'),
                ('SM Elliott', 'Sam Elliott'),
                ('JM Vince', 'James Vince'),
                ('Mohammed Shami', 'M Shami'),
                ('AB de Villiers', 'A de Villiers'),
                ('MS Dhoni', 'Mahendra Singh Dhoni'),
                ('V Kohli', 'Virat Kohli'),
                ('R Ashwin', 'Ravichandran Ashwin'),
                ('DJ Bravo', 'Dwayne Bravo'),
                ('CH Gayle', 'Chris Gayle')
            ]
        
        name_results = []
        
        start_time = time.time()
        
        for name1, name2 in test_cases:
            similarity = self.aligner.fuzzy_name_similarity(name1, name2)
            cricket_sim = self.aligner._calculate_cricket_name_similarity(name1, name2)
            
            result = {
                'name1': name1,
                'name2': name2,
                'similarity': similarity,
                'cricket_similarity': cricket_sim
            }
            
            name_results.append(result)
            
            print(f"  '{name1}' vs '{name2}': {similarity:.3f} (cricket: {cricket_sim:.3f})")
            
        total_time = time.time() - start_time
        
        # Calculate overall metrics
        similarities = [r['similarity'] for r in name_results]
        avg_similarity = np.mean(similarities)
        perfect_matches = sum(1 for s in similarities if s >= 0.95)
        good_matches = sum(1 for s in similarities if s >= 0.8)
        
        summary = {
            'total_tests': len(test_cases),
            'total_time': total_time,
            'avg_similarity': avg_similarity,
            'perfect_matches': perfect_matches,
            'good_matches': good_matches,
            'detailed_results': name_results
        }
        
        print(f"\n📊 Name Matching Summary:")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Perfect matches (≥0.95): {perfect_matches}/{len(test_cases)}")
        print(f"  Good matches (≥0.8): {good_matches}/{len(test_cases)}")
        print(f"  Total time: {total_time:.3f}s")
        
        self.results['name_matching'] = summary
        return summary
        
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        print(f"\n📋 COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'nvplay_records': len(self.aligner.nvplay_df),
                'decimal_records': len(self.aligner.decimal_df),
                'similarity_threshold': self.aligner.config.similarity_threshold,
                'fingerprint_length': self.aligner.config.fingerprint_length
            },
            'benchmark_results': self.results
        }
        
        # Save to file
        with open('match_fingerprinter_benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📄 Benchmark report saved to: match_fingerprinter_benchmark_report.json")
        
        # Print summary
        print(f"\n🏆 PERFORMANCE SUMMARY:")
        if 'fingerprint_extraction' in self.results:
            fastest_extraction = max(self.results['fingerprint_extraction'], key=lambda x: x['records_per_second'])
            print(f"  Fastest extraction: {fastest_extraction['records_per_second']:.0f} records/second")
            
        if 'similarity_calculation' in self.results:
            sim_result = self.results['similarity_calculation']
            print(f"  Similarity calculation: {sim_result['comparisons_per_second']:.0f} comparisons/second")
            
        if 'match_finding' in self.results:
            best_threshold = max(self.results['match_finding'], key=lambda x: x['num_matches'])
            print(f"  Best threshold: {best_threshold['threshold']} ({best_threshold['num_matches']} matches)")
            
        if 'name_matching' in self.results:
            name_result = self.results['name_matching']
            print(f"  Name matching accuracy: {name_result['good_matches']}/{name_result['total_tests']} good matches")
            
        return report
        
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 STARTING FULL BENCHMARK SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Initialize aligner
        init_time = self.initialize_aligner()
        
        # Run benchmarks
        self.benchmark_fingerprint_extraction()
        self.benchmark_similarity_calculation()
        self.benchmark_match_finding()
        self.benchmark_name_matching()
        
        # Generate report
        report = self.generate_benchmark_report()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 BENCHMARK COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Report saved to: match_fingerprinter_benchmark_report.json")
        
        return report

def main():
    """Main benchmark execution."""
    data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data'
    nvplay_path = f'{data_path}/nvplay_data_v3.csv'
    decimal_path = f'{data_path}/decimal_data_v3.csv'
    
    benchmark = MatchFingerprinterBenchmark(nvplay_path, decimal_path)
    report = benchmark.run_full_benchmark()
    
    return report

if __name__ == "__main__":
    main() 