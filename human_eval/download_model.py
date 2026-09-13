#!/usr/bin/env python3
"""Qwen 모델 사전 다운로드 스크립트"""
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache/hub'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache/transformers'

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model_name = "Qwen/Qwen3-8B"
print(f"🔄 {model_name} 다운로드 시작...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer 완료")
    
    config = AutoConfig.from_pretrained(model_name)
    print("✅ Config 완료")
    
    # 모델은 실제 로드하지 않고 다운로드만 확인
    print("✅ 준비 완료")
except Exception as e:
    print(f"❌ 에러: {e}")
    exit(1)
