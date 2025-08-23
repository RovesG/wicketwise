# Test script to verify Cricsheet data type fixes
import pandas as pd
import json
from cricsheet_auto_updater import get_cricsheet_updater

# Create a sample DataFrame that mimics problematic Cricsheet data
test_data = {
    'match_id': [1, 2, 3],
    'fielder_ids': [[1, 2, 3], [4, 5], [6, 7, 8, 9]],  # List columns
    'wicket_details': [
        {'type': 'caught', 'fielder': 'Smith'}, 
        {'type': 'bowled'}, 
        {'type': 'lbw'}
    ],  # Dict columns
    'team_players': [['Player1', 'Player2'], ['Player3'], ['Player4', 'Player5', 'Player6']],
    'normal_string': ['A', 'B', 'C'],
    'normal_number': [1, 2, 3]
}

df = pd.DataFrame(test_data)
print("Original DataFrame with problematic columns:")
print(df.dtypes)
print(df.head())
print()

# Test the fix
updater = get_cricsheet_updater()
fixed_df = updater._fix_data_types_for_parquet(df)

print("Fixed DataFrame:")
print(fixed_df.dtypes)
print(fixed_df.head())
print()

# Test Parquet compatibility
try:
    fixed_df.to_parquet('test_cricsheet.parquet', index=False)
    print("✅ Successfully saved to Parquet!")
    
    # Read back and verify
    read_back = pd.read_parquet('test_cricsheet.parquet')
    print("✅ Successfully read back from Parquet!")
    
    # Show that list data can be parsed back
    print("\nVerifying data integrity:")
    for i, row in read_back.iterrows():
        fielder_ids = json.loads(row['fielder_ids'])
        team_players = json.loads(row['team_players'])
        print(f"Row {i}: fielder_ids={fielder_ids}, team_players={team_players}")
    
    # Cleanup
    import os
    os.remove('test_cricsheet.parquet')
    print("\n🧹 Test cleanup complete")
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
