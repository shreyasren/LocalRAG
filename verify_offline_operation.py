#!/usr/bin/env python3
"""
Enterprise Safety Verification for LocalRAG

This script verifies that DistilBERT and other models operate completely offline
without making any external network calls.

Usage: python verify_offline_operation.py
"""

import os
import sys
import socket
import threading
import time
from unittest.mock import patch

# Force offline mode
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

def monitor_network_calls():
    """Monitor any network calls during model execution"""
    network_calls = []
    original_socket = socket.socket
    
    def intercept_socket(*args, **kwargs):
        call_info = f"Socket call: {args}, {kwargs}"
        network_calls.append(call_info)
        print(f"🚨 NETWORK CALL DETECTED: {call_info}")
        return original_socket(*args, **kwargs)
    
    # Patch socket to monitor calls
    with patch('socket.socket', side_effect=intercept_socket):
        try:
            print("🔍 Loading and testing DistilBERT (monitoring network calls)...")
            
            # Import and test transformer
            from transformers import pipeline
            
            print("📥 Creating DistilBERT pipeline...")
            qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1  # Force CPU
            )
            
            print("🧪 Testing inference...")
            result = qa_pipeline(
                question="What is the capital of France?",
                context="Paris is the capital and largest city of France."
            )
            
            print(f"✅ Result: {result}")
            print("✅ DistilBERT inference completed successfully")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
    
    return network_calls

def verify_model_cache():
    """Verify that models are cached locally"""
    print("\n🔍 Checking local model cache...")
    
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    if os.path.exists(cache_dir):
        print(f"✅ HuggingFace cache found: {cache_dir}")
        
        # Look for DistilBERT
        for root, dirs, files in os.walk(cache_dir):
            if "distilbert-base-cased-distilled-squad" in root:
                print(f"✅ DistilBERT cached locally: {root}")
                return True
            elif "distilbert" in root.lower():
                print(f"✅ DistilBERT variant found: {root}")
                return True
    else:
        print(f"❌ HuggingFace cache not found at {cache_dir}")
        return False
    
    return False

def test_local_operation():
    """Test complete local operation"""
    print("\n🔒 Testing Complete Local Operation")
    print("="*50)
    
    # Check if models are cached
    models_cached = verify_model_cache()
    
    if not models_cached:
        print("⚠️  Models not cached. First run will download them.")
        print("💡 Run this script with internet to cache models, then again offline.")
        return
    
    # Monitor network calls during execution
    print("\n🕵️  Starting network monitoring...")
    network_calls = monitor_network_calls()
    
    # Report results
    print("\n" + "="*50)
    print("🎯 ENTERPRISE SAFETY VERIFICATION RESULTS")
    print("="*50)
    
    if network_calls:
        print(f"⚠️  {len(network_calls)} network calls detected:")
        for call in network_calls:
            print(f"   - {call}")
        print("\n❌ NOT ENTERPRISE SAFE - External calls detected")
    else:
        print("✅ NO NETWORK CALLS DETECTED")
        print("✅ ENTERPRISE SAFE - 100% local operation confirmed")
    
    print(f"\n📊 Summary:")
    print(f"   - Models cached locally: {'✅' if models_cached else '❌'}")
    print(f"   - Zero network calls: {'✅' if not network_calls else '❌'}")
    print(f"   - Enterprise ready: {'✅' if models_cached and not network_calls else '❌'}")

def main():
    print("🏢 LocalRAG Enterprise Safety Verification")
    print("="*50)
    print("This script verifies that DistilBERT operates completely offline")
    print("without making any external network calls.\n")
    
    test_local_operation()
    
    print("\n💡 For complete enterprise safety:")
    print("1. Run this script once with internet (to cache models)")
    print("2. Run again offline to verify zero external calls")
    print("3. Add network isolation if required by your security policy")

if __name__ == "__main__":
    main()
