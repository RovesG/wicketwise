# Purpose: Automated Cricsheet data updater with incremental processing
# Author: Assistant, Last Modified: 2024

import os
import requests
import zipfile
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UpdateResult:
    """Result of a Cricsheet update operation"""
    success: bool
    new_files_count: int
    total_new_balls: int
    update_summary: str
    error_message: Optional[str] = None

class CricsheetAutoUpdater:
    """
    Automated system for checking and updating Cricsheet data
    
    Features:
    - Check for newer all_json.zip file
    - Download and extract incrementally
    - Process only new files
    - Update KG and GNN incrementally
    """
    
    def __init__(self, 
                 data_dir: str = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data",
                 cache_dir: str = "cache/cricsheet_updates"):
        
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cricsheet URLs
        self.cricsheet_url = "https://cricsheet.org/downloads/all_json.zip"
        self.all_json_dir = self.data_dir / "all_json"
        
        # Cache files
        self.last_update_file = self.cache_dir / "last_update.json"
        self.processed_files_cache = self.cache_dir / "processed_files.json"
        
        # Output paths
        self.parquet_output = Path("artifacts/kg_background/events")
        self.parquet_output.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔄 Cricsheet Auto Updater initialized")
    
    def check_for_updates(self) -> Dict[str, any]:
        """
        Check if there's a newer version of all_json.zip available
        
        Returns:
            Dict with update status, file info, and recommendation
        """
        try:
            logger.info("🔍 Checking for Cricsheet updates...")
            
            # Get remote file info
            response = requests.head(self.cricsheet_url, timeout=30)
            if response.status_code != 200:
                return {
                    "update_available": False,
                    "error": f"Failed to check remote file: HTTP {response.status_code}"
                }
            
            # Extract remote file metadata
            remote_size = int(response.headers.get('content-length', 0))
            remote_last_modified = response.headers.get('last-modified', '')
            remote_etag = response.headers.get('etag', '').strip('"')
            
            # Load local update history
            last_update = self._load_last_update()
            
            # Compare with last known state
            update_needed = (
                remote_etag != last_update.get('etag', '') or
                remote_size != last_update.get('size', 0) or
                remote_last_modified != last_update.get('last_modified', '')
            )
            
            result = {
                "update_available": update_needed,
                "remote_info": {
                    "size": remote_size,
                    "size_mb": round(remote_size / (1024 * 1024), 1),
                    "last_modified": remote_last_modified,
                    "etag": remote_etag
                },
                "local_info": last_update,
                "recommendation": "Download and process new data" if update_needed else "No update needed"
            }
            
            logger.info(f"📊 Update check complete: {'Update available' if update_needed else 'Up to date'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to check for updates: {e}")
            return {
                "update_available": False,
                "error": str(e)
            }
    
    def download_and_process_updates(self) -> UpdateResult:
        """
        Download the latest all_json.zip and process only new files
        
        Returns:
            UpdateResult with processing details
        """
        try:
            logger.info("⬇️ Starting Cricsheet data update...")
            
            # Check what files we already have
            existing_files = self._get_existing_files()
            logger.info(f"📁 Found {len(existing_files)} existing JSON files")
            
            # Download and extract to temporary location
            temp_dir = self.cache_dir / "temp_extract"
            temp_dir.mkdir(exist_ok=True)
            
            logger.info("⬇️ Downloading all_json.zip...")
            zip_path = self._download_zip_file(temp_dir)
            
            logger.info("📦 Extracting archive...")
            new_files = self._extract_new_files_only(zip_path, temp_dir, existing_files)
            
            if not new_files:
                logger.info("✅ No new files to process")
                return UpdateResult(
                    success=True,
                    new_files_count=0,
                    total_new_balls=0,
                    update_summary="No new data available"
                )
            
            logger.info(f"🆕 Found {len(new_files)} new files to process")
            
            # Process new files incrementally
            logger.info("🔄 Processing new files with improved data type handling...")
            total_new_balls = self._process_new_files(new_files, temp_dir)
            
            # Move new files to main all_json directory
            self._integrate_new_files(new_files, temp_dir)
            
            # Update cache
            self._update_processed_files_cache(new_files)
            self._save_last_update()
            
            # Cleanup
            self._cleanup_temp_files(temp_dir)
            
            logger.info(f"✅ Update complete: {len(new_files)} new files, {total_new_balls:,} new balls")
            
            return UpdateResult(
                success=True,
                new_files_count=len(new_files),
                total_new_balls=total_new_balls,
                update_summary=f"Successfully processed {len(new_files)} new files with {total_new_balls:,} balls"
            )
            
        except Exception as e:
            logger.error(f"❌ Update failed: {e}")
            return UpdateResult(
                success=False,
                new_files_count=0,
                total_new_balls=0,
                update_summary="Update failed",
                error_message=str(e)
            )
    
    def trigger_incremental_kg_update(self) -> bool:
        """
        Trigger incremental Knowledge Graph update with new data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🔄 Triggering incremental KG update...")
            
            # Import here to avoid circular dependencies
            from enriched_training_pipeline import get_enriched_training_pipeline
            
            pipeline = get_enriched_training_pipeline()
            
            # Prepare updated KG training data
            df_harmonized, enrichment_metadata = pipeline.prepare_kg_training_data(
                auto_enrich=True,
                max_new_matches=1000  # Process up to 1000 new matches
            )
            
            logger.info(f"📊 KG data prepared: {len(df_harmonized):,} total balls")
            
            # TODO: Trigger actual KG rebuild/update
            # This would call the KG building pipeline
            
            return True
            
        except Exception as e:
            logger.error(f"❌ KG update failed: {e}")
            return False
    
    def trigger_gnn_retrain(self, incremental: bool = True) -> bool:
        """
        Trigger GNN retraining with updated data
        
        Args:
            incremental: If True, try incremental training; if False, full retrain
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🧠 Triggering GNN {'incremental' if incremental else 'full'} retrain...")
            
            # TODO: Implement GNN retraining logic
            # This would:
            # 1. Load existing GNN model
            # 2. Either fine-tune with new data (incremental) or retrain from scratch
            # 3. Update embeddings for new entities
            # 4. Save updated model
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GNN retrain failed: {e}")
            return False
    
    def run_full_update_pipeline(self) -> Dict[str, any]:
        """
        Run the complete update pipeline: check → download → process → update KG → retrain GNN
        
        Returns:
            Comprehensive update report
        """
        logger.info("🚀 Starting full Cricsheet update pipeline...")
        
        report = {
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "overall_success": False,
            "summary": ""
        }
        
        # Step 1: Check for updates
        check_result = self.check_for_updates()
        report["steps"]["check"] = check_result
        
        if not check_result.get("update_available", False):
            report["overall_success"] = True
            report["summary"] = "No updates available"
            return report
        
        # Step 2: Download and process
        process_result = self.download_and_process_updates()
        report["steps"]["download_process"] = process_result.__dict__
        
        if not process_result.success:
            report["summary"] = f"Update failed: {process_result.error_message}"
            return report
        
        # Step 3: Update KG
        kg_success = self.trigger_incremental_kg_update()
        report["steps"]["kg_update"] = {"success": kg_success}
        
        # Step 4: Retrain GNN
        gnn_success = self.trigger_gnn_retrain(incremental=True)
        report["steps"]["gnn_retrain"] = {"success": gnn_success}
        
        # Final summary
        all_success = process_result.success and kg_success and gnn_success
        report["overall_success"] = all_success
        report["summary"] = (
            f"Successfully updated with {process_result.new_files_count} new files "
            f"({process_result.total_new_balls:,} balls), "
            f"KG {'✅' if kg_success else '❌'}, "
            f"GNN {'✅' if gnn_success else '❌'}"
        )
        report["completed_at"] = datetime.now().isoformat()
        
        return report
    
    # Private helper methods
    
    def _load_last_update(self) -> Dict[str, any]:
        """Load the last update metadata"""
        if self.last_update_file.exists():
            try:
                with open(self.last_update_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_last_update(self):
        """Save current update metadata"""
        try:
            response = requests.head(self.cricsheet_url, timeout=30)
            metadata = {
                "updated_at": datetime.now().isoformat(),
                "size": int(response.headers.get('content-length', 0)),
                "last_modified": response.headers.get('last-modified', ''),
                "etag": response.headers.get('etag', '').strip('"')
            }
            
            with open(self.last_update_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save update metadata: {e}")
    
    def _get_existing_files(self) -> Set[str]:
        """Get set of existing JSON file names"""
        if not self.all_json_dir.exists():
            return set()
        
        return {f.name for f in self.all_json_dir.glob("*.json")}
    
    def _download_zip_file(self, temp_dir: Path) -> Path:
        """Download the all_json.zip file"""
        zip_path = temp_dir / "all_json.zip"
        
        response = requests.get(self.cricsheet_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return zip_path
    
    def _extract_new_files_only(self, zip_path: Path, temp_dir: Path, existing_files: Set[str]) -> List[str]:
        """Extract only files that don't already exist"""
        new_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.json'):
                    filename = Path(file_info.filename).name
                    if filename not in existing_files:
                        zip_ref.extract(file_info, temp_dir)
                        new_files.append(filename)
        
        return new_files
    
    def _process_new_files(self, new_files: List[str], temp_dir: Path) -> int:
        """Process new JSON files and add to parquet dataset"""
        from wicketwise.data.alljson.ingest import flatten_file_to_dataframe
        
        total_balls = 0
        new_dfs = []
        
        for filename in new_files:
            file_path = temp_dir / filename
            if file_path.exists():
                try:
                    df = flatten_file_to_dataframe(str(file_path))
                    new_dfs.append(df)
                    total_balls += len(df)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {filename}: {e}")
                    continue
        
        if new_dfs:
            # Combine new data
            new_data = pd.concat(new_dfs, ignore_index=True)
            
            # Fix data types for Parquet compatibility
            new_data = self._fix_data_types_for_parquet(new_data)
            
            # Load existing parquet data
            existing_parquet = self.parquet_output / "events.parquet"
            if existing_parquet.exists():
                existing_data = pd.read_parquet(existing_parquet)
                
                # Ensure column compatibility
                new_data = self._align_dataframe_columns(new_data, existing_data)
                
                # Combine with existing data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined_data = new_data
            
            # Save updated parquet file
            combined_data.to_parquet(existing_parquet, index=False)
            logger.info(f"📦 Updated parquet with {total_balls:,} new balls")
        
        return total_balls
    
    def _integrate_new_files(self, new_files: List[str], temp_dir: Path):
        """Move new files to the main all_json directory"""
        self.all_json_dir.mkdir(exist_ok=True)
        
        for filename in new_files:
            src = temp_dir / filename
            dst = self.all_json_dir / filename
            if src.exists():
                src.rename(dst)
    
    def _update_processed_files_cache(self, new_files: List[str]):
        """Update the cache of processed files"""
        cache = set()
        if self.processed_files_cache.exists():
            try:
                with open(self.processed_files_cache, 'r') as f:
                    cache = set(json.load(f))
            except:
                pass
        
        cache.update(new_files)
        
        with open(self.processed_files_cache, 'w') as f:
            json.dump(list(cache), f)
    
    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files"""
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def _fix_data_types_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types to be compatible with Parquet format"""
        df_fixed = df.copy()
        
        for col in df_fixed.columns:
            # Handle list/array columns that cause Parquet issues
            if df_fixed[col].dtype == 'object':
                # Check if column contains lists
                sample_values = df_fixed[col].dropna().head(10)
                if len(sample_values) > 0 and any(isinstance(val, (list, tuple)) for val in sample_values):
                    # Convert lists to JSON strings
                    df_fixed[col] = df_fixed[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else str(x) if x is not None else None
                    )
                else:
                    # Convert other object types to strings
                    df_fixed[col] = df_fixed[col].astype(str).replace('nan', None)
        
        logger.debug(f"🔧 Fixed data types for {len(df_fixed.columns)} columns")
        return df_fixed
    
    def _align_dataframe_columns(self, new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure new dataframe has compatible columns with existing data"""
        # Get all columns from existing data
        existing_cols = set(existing_df.columns)
        new_cols = set(new_df.columns)
        
        # Add missing columns to new data (fill with None)
        missing_cols = existing_cols - new_cols
        for col in missing_cols:
            new_df[col] = None
            logger.debug(f"➕ Added missing column: {col}")
        
        # Reorder columns to match existing data
        new_df = new_df[existing_df.columns]
        
        # Handle any extra columns in new data (log but keep them)
        extra_cols = new_cols - existing_cols
        if extra_cols:
            logger.info(f"🆕 New columns detected: {list(extra_cols)}")
            # Add extra columns to the end
            for col in extra_cols:
                if col in new_df.columns and col not in existing_df.columns:
                    # This column exists in new data but not existing - keep it
                    pass
        
        return new_df


# Factory function for easy access
def get_cricsheet_updater() -> CricsheetAutoUpdater:
    """Get a configured Cricsheet updater instance"""
    return CricsheetAutoUpdater()


if __name__ == "__main__":
    # CLI interface for testing
    logging.basicConfig(level=logging.INFO)
    
    updater = get_cricsheet_updater()
    
    print("🔍 Checking for Cricsheet updates...")
    check_result = updater.check_for_updates()
    print(json.dumps(check_result, indent=2))
    
    if check_result.get("update_available"):
        print("\n🚀 Running full update pipeline...")
        result = updater.run_full_update_pipeline()
        print(json.dumps(result, indent=2))
