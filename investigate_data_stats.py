#!/usr/bin/env python3
"""
Investigate Data Statistics - Check actual KG and T20 data
"""

import pickle
import pandas as pd
from pathlib import Path
import sys

def investigate_data_stats():
    """Investigate actual data statistics"""
    
    print("🔍 DATA STATISTICS INVESTIGATION")
    print("=" * 50)
    
    # 1. Check T20 Dataset
    print("\n📊 T20 DATASET ANALYSIS:")
    print("-" * 30)
    
    t20_path = Path("/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/joined_ball_by_ball_data.csv")
    
    if t20_path.exists():
        try:
            df = pd.read_csv(t20_path)
            print(f"✅ T20 Dataset loaded: {len(df):,} rows")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Check for match information
            if 'match_id' in df.columns:
                unique_matches = df['match_id'].nunique()
                print(f"🏏 Unique matches (match_id): {unique_matches:,}")
            elif 'date' in df.columns and 'home_team' in df.columns:
                # Create match identifier from available columns
                match_cols = []
                for col in ['date', 'home_team', 'away_team', 'competition']:
                    if col in df.columns:
                        match_cols.append(col)
                
                if match_cols:
                    match_df = df[match_cols].drop_duplicates()
                    unique_matches = len(match_df)
                    print(f"🏏 Unique matches (estimated from {'+'.join(match_cols)}): {unique_matches:,}")
            
            # Check venues
            venue_cols = [col for col in df.columns if 'venue' in col.lower() or 'ground' in col.lower()]
            if venue_cols:
                for col in venue_cols:
                    unique_venues = df[col].nunique()
                    print(f"🏟️ Unique venues ({col}): {unique_venues:,}")
            
            print(f"📈 Sample data:")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"❌ Error loading T20 dataset: {e}")
    else:
        print(f"❌ T20 dataset not found at: {t20_path}")
    
    # 2. Check Knowledge Graph
    print(f"\n🕸️ KNOWLEDGE GRAPH ANALYSIS:")
    print("-" * 30)
    
    kg_path = Path("models/unified_cricket_kg.pkl")
    
    if kg_path.exists():
        try:
            with open(kg_path, 'rb') as f:
                kg_data = pickle.load(f)
            
            print(f"✅ KG loaded successfully")
            print(f"📋 KG type: {type(kg_data)}")
            
            if isinstance(kg_data, dict):
                print(f"🔑 KG keys: {list(kg_data.keys())}")
                
                # Check different data sections
                for key, value in kg_data.items():
                    if hasattr(value, '__len__'):
                        try:
                            length = len(value)
                            print(f"   {key}: {length:,} items")
                            
                            # If it's a DataFrame, show more details
                            if hasattr(value, 'columns'):
                                print(f"     Columns: {list(value.columns)}")
                                if len(value) > 0:
                                    print(f"     Sample: {value.iloc[0].to_dict()}")
                        except:
                            print(f"   {key}: {type(value)}")
                    else:
                        print(f"   {key}: {type(value)}")
            
        except Exception as e:
            print(f"❌ Error loading KG: {e}")
    else:
        print(f"❌ KG not found at: {kg_path}")
    
    # 3. Check old KG for comparison
    print(f"\n🕸️ OLD KNOWLEDGE GRAPH ANALYSIS:")
    print("-" * 30)
    
    old_kg_path = Path("models/cricket_knowledge_graph.pkl")
    
    if old_kg_path.exists():
        try:
            with open(old_kg_path, 'rb') as f:
                old_kg_data = pickle.load(f)
            
            print(f"✅ Old KG loaded successfully")
            print(f"📋 Old KG type: {type(old_kg_data)}")
            
            if isinstance(old_kg_data, dict):
                print(f"🔑 Old KG keys: {list(old_kg_data.keys())}")
                
                for key, value in old_kg_data.items():
                    if hasattr(value, '__len__'):
                        try:
                            length = len(value)
                            print(f"   {key}: {length:,} items")
                        except:
                            print(f"   {key}: {type(value)}")
            
        except Exception as e:
            print(f"❌ Error loading old KG: {e}")
    else:
        print(f"❌ Old KG not found at: {old_kg_path}")

if __name__ == "__main__":
    investigate_data_stats()
