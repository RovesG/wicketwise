# Purpose: Debug OpenAI model responses for enrichment
# Author: WicketWise Team, Last Modified: 2025-08-26

import os
import json
from openai import OpenAI

def test_openai_models():
    """Test different OpenAI models to find the most reliable one"""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    models_to_test = ['gpt-4o', 'gpt-4o-mini', 'gpt-5-mini']
    
    test_prompt = """Provide cricket match data in JSON format:
Date: 2021-08-28
Venue: Warner Park  
Teams: Barbados Royals vs Jamaica Tallawahs
Competition: Caribbean Premier League

Include weather conditions and venue coordinates if available. Return only valid JSON."""

    print("🧪 Testing OpenAI Models for Cricket Enrichment")
    print("=" * 60)
    
    for model in models_to_test:
        print(f"\n🔍 Testing {model}:")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'You are a cricket data expert. Return only valid JSON, no markdown formatting.'},
                    {'role': 'user', 'content': test_prompt}
                ],
                max_completion_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            print(f"   📏 Response length: {len(content)}")
            
            if not content:
                print("   ❌ Empty response")
                continue
            
            print(f"   📝 First 150 chars: {content[:150]}...")
            
            # Clean up potential markdown formatting
            cleaned_content = content
            if content.startswith('```'):
                lines = content.split('\n')
                json_lines = []
                in_json = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                
                cleaned_content = '\n'.join(json_lines)
            
            # Try to parse JSON
            try:
                parsed = json.loads(cleaned_content)
                print(f"   ✅ JSON valid! Keys: {list(parsed.keys())}")
                
                # Check for important fields
                has_weather = 'weather' in parsed or 'weather_hourly' in parsed
                has_venue = 'venue' in parsed
                has_coords = False
                
                if has_venue and isinstance(parsed['venue'], dict):
                    venue = parsed['venue']
                    has_coords = (venue.get('latitude', 0) != 0 or venue.get('longitude', 0) != 0)
                
                print(f"   🌤️ Weather data: {'✅' if has_weather else '❌'}")
                print(f"   🏟️ Venue data: {'✅' if has_venue else '❌'}")
                print(f"   📍 Coordinates: {'✅' if has_coords else '❌'}")
                
                if has_weather or has_coords:
                    print(f"   🎉 {model} works well for enrichment!")
                    return model, parsed
                
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing failed: {e}")
                print(f"   📄 Cleaned content: {cleaned_content[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Model call failed: {e}")
    
    print("\n⚠️ No model produced reliable enrichment data")
    return None, None

if __name__ == "__main__":
    working_model, sample_data = test_openai_models()
    
    if working_model:
        print(f"\n🎯 Recommended model: {working_model}")
        print(f"\n📊 Sample enriched data:")
        print(json.dumps(sample_data, indent=2)[:500] + "...")
    else:
        print("\n❌ Need to investigate API access or model availability")
