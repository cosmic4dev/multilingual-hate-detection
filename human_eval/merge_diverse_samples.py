#!/usr/bin/env python3
"""
기존 30개 샘플과 새로운 diverse 27개 샘플을 합쳐서
다양성이 최대화된 30개를 만듭니다.
"""

import csv
import os

def analyze_pattern(text):
    """텍스트의 패턴 분석"""
    text_lower = text.lower()
    patterns = []
    
    if 'you\'re such a' in text_lower or 'you\'re such an' in text_lower:
        patterns.append('such_pattern')
    if 'you\'re being' in text_lower or 'you\'re acting' in text_lower:
        patterns.append('being_pattern')
    if text.strip().startswith('go ') or text.strip().startswith('shut') or text.strip().startswith('stop'):
        patterns.append('명령문')
    if '?' in text:
        patterns.append('질문형')
    if '!' in text and not text.strip().endswith('.'):
        patterns.append('감탄문')
    if 'fuck' in text_lower:
        patterns.append('fuck_포함')
    
    return patterns if patterns else ['기타']

def load_and_prepare_samples():
    # 기존 30개
    with open('/root/multilingual-hate-detection/human_eval/human_eval_ready_qwen25_30_spans.csv', 'r', encoding='utf-8') as f:
        existing = list(csv.DictReader(f))
    
    # diverse 27개
    with open('/root/multilingual-hate-detection/human_eval/diverse_samples_with_spans.csv', 'r', encoding='utf-8') as f:
        diverse = list(csv.DictReader(f))
    
    # diverse 샘플에 타입 정보 추가 (이미 있음)
    for row in diverse:
        row['is_diverse'] = True
        row['patterns'] = analyze_pattern(row['original_text'])
    
    for row in existing:
        row['is_diverse'] = False
        row['patterns'] = analyze_pattern(row['original_text'])
    
    return existing, diverse

def select_diverse_30():
    existing, diverse = load_and_prepare_samples()
    
    # 전략: diverse 샘플을 우선적으로 선택하고, 
    # 부족한 부분은 기존 샘플로 보완
    selected = []
    
    # 1. diverse 샘플 모두 추가 (27개)
    selected.extend(diverse)
    
    # 2. 기존 샘플 중에서 diverse와 다른 패턴을 가진 것들 선택
    # diverse에 없는 패턴을 찾기
    diverse_patterns = set()
    for row in diverse:
        diverse_patterns.update(row['patterns'])
    
    # 기존 샘플 중 diverse와 다른 패턴을 가진 것 찾기
    remaining_from_existing = []
    for row in existing:
        row_patterns = set(row['patterns'])
        # diverse와 겹치지 않는 패턴이 있으면 우선순위 높음
        overlap = len(row_patterns & diverse_patterns)
        row['pattern_overlap'] = overlap
        remaining_from_existing.append(row)
    
    # 패턴 겹침이 적은 순으로 정렬
    remaining_from_existing.sort(key=lambda x: x['pattern_overlap'])
    
    # 3개 더 추가 (30개 완성) - strong/mild 균형 고려
    # 현재 strong/mild 개수 확인
    current_strong = sum(1 for s in selected if s['toxicity_strength'] == 'strong')
    current_mild = sum(1 for s in selected if s['toxicity_strength'] == 'mild')
    
    # 목표: strong 15, mild 15
    needed_strong = max(0, 15 - current_strong)
    needed_mild = max(0, 15 - current_mild)
    
    # strong 우선 추가
    for row in remaining_from_existing:
        if len(selected) >= 30 or (needed_strong == 0 and needed_mild == 0):
            break
        if row['toxicity_strength'] == 'strong' and needed_strong > 0:
            text_lower = row['original_text'].lower()
            is_duplicate = False
            for selected_row in selected:
                selected_text_lower = selected_row['original_text'].lower()
                if text_lower == selected_text_lower or (len(text_lower) < 50 and text_lower in selected_text_lower):
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(row)
                needed_strong -= 1
    
    # mild 추가
    for row in remaining_from_existing:
        if len(selected) >= 30 or (needed_mild == 0):
            break
        if row['toxicity_strength'] == 'mild' and needed_mild > 0 and row not in selected:
            text_lower = row['original_text'].lower()
            is_duplicate = False
            for selected_row in selected:
                selected_text_lower = selected_row['original_text'].lower()
                if text_lower == selected_text_lower or (len(text_lower) < 50 and text_lower in selected_text_lower):
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(row)
                needed_mild -= 1
    
    # 여전히 부족하면 나머지로 채우기
    for row in remaining_from_existing:
        if len(selected) >= 30:
            break
        if row not in selected:
            text_lower = row['original_text'].lower()
            is_duplicate = False
            for selected_row in selected:
                selected_text_lower = selected_row['original_text'].lower()
                if text_lower == selected_text_lower:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(row)
    
    # 정확히 30개로 제한
    selected = selected[:30]
    
    # strong/mild 균형 조정 (가능한 한)
    strong_count = sum(1 for s in selected if s['toxicity_strength'] == 'strong')
    mild_count = sum(1 for s in selected if s['toxicity_strength'] == 'mild')
    
    print(f"\n최종 선택: {len(selected)}개")
    print(f"  Strong: {strong_count}개")
    print(f"  Mild: {mild_count}개")
    print(f"  Diverse 샘플: {sum(1 for s in selected if s.get('is_diverse', False))}개")
    print(f"  기존 샘플: {sum(1 for s in selected if not s.get('is_diverse', False))}개")
    
    # 패턴 분포 확인
    all_patterns = {}
    for row in selected:
        for pattern in row['patterns']:
            all_patterns[pattern] = all_patterns.get(pattern, 0) + 1
    
    print(f"\n패턴 분포:")
    for pattern, count in sorted(all_patterns.items()):
        print(f"  {pattern}: {count}개")
    
    return selected

def main():
    selected = select_diverse_30()
    
    # output_unguided와 output_span_guided가 있는지 확인
    # diverse 샘플은 아직 생성이 안 되어있을 수 있음
    output_file = '/root/multilingual-hate-detection/human_eval/human_eval_ready_qwen25_merged_30.csv'
    
    # 기존 생성 결과 로드
    with open('/root/multilingual-hate-detection/human_eval/human_eval_ready_qwen25_30_spans.csv', 'r', encoding='utf-8') as f:
        existing_results = {row['sample_id']: row for row in csv.DictReader(f)}
    
    with open('/root/multilingual-hate-detection/human_eval/human_eval_ready_qwen25_diverse_27.csv', 'r', encoding='utf-8') as f:
        diverse_results = {row['sample_id']: row for row in csv.DictReader(f)}
    
    # 최종 결과 작성
    final_rows = []
    for row in selected:
        sample_id = row['sample_id']
        
        # 기존 결과 또는 diverse 결과에서 찾기
        if sample_id in existing_results:
            result_row = existing_results[sample_id]
        elif sample_id in diverse_results:
            result_row = diverse_results[sample_id]
        else:
            # 결과가 없으면 기본값
            result_row = {
                'sample_id': sample_id,
                'original_text': row['original_text'],
                'toxicity_strength': row['toxicity_strength'],
                'output_unguided': '',  # 나중에 생성 필요
                'output_span_guided': '',  # 나중에 생성 필요
            }
        
        final_rows.append({
            'sample_id': result_row.get('sample_id', sample_id),
            'language': result_row.get('language', 'EN'),
            'toxicity_strength': result_row.get('toxicity_strength', row['toxicity_strength']),
            'original_text': result_row.get('original_text', row['original_text']),
            'detected_spans_text': result_row.get('detected_spans_text', ''),
            'detected_spans_intensity': result_row.get('detected_spans_intensity', ''),
            'span_empty': result_row.get('span_empty', 'False'),
            'output_unguided': result_row.get('output_unguided', ''),
            'output_span_guided': result_row.get('output_span_guided', ''),
        })
    
    # CSV 저장
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'sample_id', 'language', 'toxicity_strength', 'original_text',
            'detected_spans_text', 'detected_spans_intensity', 'span_empty',
            'output_unguided', 'output_span_guided'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    
    print(f"\n✅ 저장 완료: {output_file}")
    
    # 빈 output이 있는지 확인
    empty_outputs = [r for r in final_rows if not r.get('output_unguided') or not r.get('output_span_guided')]
    if empty_outputs:
        print(f"\n⚠️  {len(empty_outputs)}개 샘플의 output이 비어있습니다. 재생성 필요합니다.")

if __name__ == '__main__':
    main()

