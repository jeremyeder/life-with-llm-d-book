#!/usr/bin/env python3
"""
Complexity Analysis Tool
Analyzes technical complexity and appropriateness for different persona levels.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from collections import Counter

class ComplexityAnalyzer:
    """Analyzes technical complexity of documentation content."""
    
    def __init__(self):
        self.technical_terms = self._load_technical_terms()
        self.complexity_indicators = self._load_complexity_indicators()
    
    def _load_technical_terms(self) -> Dict[str, int]:
        """Load technical terms with complexity weights."""
        return {
            # Basic terms (weight 1)
            'docker': 1, 'container': 1, 'image': 1, 'pod': 1, 'service': 1,
            'deployment': 1, 'namespace': 1, 'kubectl': 1, 'yaml': 1,
            
            # Intermediate terms (weight 2)
            'kubernetes': 2, 'openshift': 2, 'helm': 2, 'ingress': 2,
            'configmap': 2, 'secret': 2, 'volume': 2, 'pvc': 2,
            'scaling': 2, 'monitoring': 2, 'logging': 2,
            
            # Advanced terms (weight 3)
            'rbac': 3, 'networkpolicy': 3, 'psp': 3, 'admission': 3,
            'webhook': 3, 'operator': 3, 'crd': 3, 'controller': 3,
            'scheduler': 3, 'etcd': 3, 'cni': 3, 'csi': 3,
            
            # ML/AI specific (weight varies)
            'inference': 2, 'model': 1, 'gpu': 2, 'nvidia': 2,
            'vllm': 3, 'llm': 2, 'tokenization': 3, 'quantization': 3,
            'attention': 3, 'transformer': 3, 'embedding': 3,
            
            # Performance/optimization (weight 3)
            'optimization': 3, 'tuning': 3, 'profiling': 3,
            'benchmarking': 3, 'scaling': 2, 'autoscaling': 2,
            'hpa': 3, 'vpa': 3, 'cluster-autoscaler': 3
        }
    
    def _load_complexity_indicators(self) -> Dict[str, int]:
        """Load complexity indicators with weights."""
        return {
            # Procedural complexity
            'step-by-step': 1,
            'tutorial': 1,
            'example': 1,
            'walkthrough': 1,
            
            # Conceptual complexity
            'architecture': 2,
            'design': 2,
            'pattern': 2,
            'principle': 2,
            
            # Implementation complexity
            'advanced': 3,
            'custom': 3,
            'extend': 3,
            'implement': 3,
            'integrate': 3,
            
            # Operational complexity
            'troubleshooting': 3,
            'debugging': 3,
            'performance': 3,
            'optimization': 3,
            'monitoring': 2,
            'alerting': 2
        }
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content complexity across multiple dimensions."""
        words = content.lower().split()
        
        # Technical term analysis
        technical_analysis = self._analyze_technical_terms(content)
        
        # Structural complexity
        structural_analysis = self._analyze_structure(content)
        
        # Code complexity
        code_analysis = self._analyze_code_blocks(content)
        
        # Procedural complexity
        procedural_analysis = self._analyze_procedures(content)
        
        # Calculate overall complexity score
        overall_score = self._calculate_overall_complexity(
            technical_analysis, structural_analysis, code_analysis, procedural_analysis
        )
        
        return {
            'overall_score': overall_score,
            'technical_complexity': technical_analysis,
            'structural_complexity': structural_analysis,
            'code_complexity': code_analysis,
            'procedural_complexity': procedural_analysis,
            'readability_metrics': self._calculate_readability(content)
        }
    
    def _analyze_technical_terms(self, content: str) -> Dict[str, Any]:
        """Analyze technical term complexity."""
        content_lower = content.lower()
        found_terms = {}
        complexity_sum = 0
        
        for term, weight in self.technical_terms.items():
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower))
            if count > 0:
                found_terms[term] = {
                    'count': count,
                    'weight': weight,
                    'contribution': count * weight
                }
                complexity_sum += count * weight
        
        return {
            'total_score': complexity_sum,
            'unique_terms': len(found_terms),
            'terms': found_terms,
            'average_weight': complexity_sum / max(1, sum(t['count'] for t in found_terms.values()))
        }
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze structural complexity."""
        lines = content.split('\n')
        
        # Count heading levels
        headings = []
        for line in lines:
            if line.strip().startswith('#'):
                level = len(line.split()[0])
                headings.append(level)
        
        # Calculate nesting complexity
        max_depth = max(headings) if headings else 0
        depth_variety = len(set(headings)) if headings else 0
        
        # Count lists and nested structures
        list_items = len([line for line in lines if re.match(r'^\s*[-*+]\s', line)])
        nested_lists = len([line for line in lines if re.match(r'^\s{2,}[-*+]\s', line)])
        
        return {
            'heading_depth': max_depth,
            'heading_variety': depth_variety,
            'total_headings': len(headings),
            'list_items': list_items,
            'nested_lists': nested_lists,
            'structural_score': min(10, max_depth + depth_variety + (nested_lists * 2))
        }
    
    def _analyze_code_blocks(self, content: str) -> Dict[str, Any]:
        """Analyze code block complexity."""
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        
        total_lines = 0
        languages = []
        complexity_scores = []
        
        for language, code in matches:
            languages.append(language or 'unknown')
            lines = code.strip().split('\n')
            total_lines += len(lines)
            
            # Calculate code complexity
            complexity = self._calculate_code_complexity(code, language)
            complexity_scores.append(complexity)
        
        return {
            'total_blocks': len(matches),
            'total_lines': total_lines,
            'languages': list(set(languages)),
            'average_block_size': total_lines / max(1, len(matches)),
            'complexity_scores': complexity_scores,
            'average_complexity': sum(complexity_scores) / max(1, len(complexity_scores))
        }
    
    def _calculate_code_complexity(self, code: str, language: str) -> int:
        """Calculate complexity score for a code block."""
        lines = code.strip().split('\n')
        complexity = 0
        
        # Base complexity from line count
        complexity += len(lines) * 0.1
        
        # Language-specific complexity
        if language in ['python', 'py']:
            complexity += self._python_complexity(code)
        elif language in ['yaml', 'yml']:
            complexity += self._yaml_complexity(code)
        elif language in ['bash', 'shell', 'sh']:
            complexity += self._bash_complexity(code)
        
        return min(10, int(complexity))
    
    def _python_complexity(self, code: str) -> int:
        """Calculate Python-specific complexity."""
        complexity = 0
        
        # Control structures
        complexity += len(re.findall(r'\b(if|for|while|try|except|with|def|class)\b', code))
        
        # Imports (especially complex ones)
        complexity += len(re.findall(r'import\s+\w+', code)) * 0.5
        complexity += len(re.findall(r'from\s+\w+\s+import', code)) * 0.3
        
        # Function calls
        complexity += len(re.findall(r'\w+\(', code)) * 0.2
        
        return int(complexity)
    
    def _yaml_complexity(self, code: str) -> int:
        """Calculate YAML-specific complexity."""
        complexity = 0
        
        # Nesting levels
        lines = code.split('\n')
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        complexity += max_indent / 2
        
        # Array items
        complexity += len(re.findall(r'^\s*-\s', code, re.MULTILINE)) * 0.1
        
        return int(complexity)
    
    def _bash_complexity(self, code: str) -> int:
        """Calculate Bash-specific complexity."""
        complexity = 0
        
        # Command complexity
        complexity += len(re.findall(r'\|', code)) * 0.5  # Pipes
        complexity += len(re.findall(r'&&|\|\|', code)) * 0.3  # Logical operators
        complexity += len(re.findall(r'if\b|for\b|while\b|case\b', code)) * 1  # Control structures
        
        return int(complexity)
    
    def _analyze_procedures(self, content: str) -> Dict[str, Any]:
        """Analyze procedural complexity."""
        # Count numbered steps
        numbered_steps = len(re.findall(r'^\d+\.', content, re.MULTILINE))
        
        # Count procedural indicators
        procedural_indicators = 0
        for indicator in ['first', 'then', 'next', 'finally', 'step']:
            procedural_indicators += len(re.findall(r'\b' + indicator + r'\b', content, re.IGNORECASE))
        
        # Count decision points
        decision_points = len(re.findall(r'\b(if|when|depending|choose|select|option)\b', content, re.IGNORECASE))
        
        return {
            'numbered_steps': numbered_steps,
            'procedural_indicators': procedural_indicators,
            'decision_points': decision_points,
            'procedural_score': min(10, numbered_steps + (procedural_indicators * 0.5) + decision_points)
        }
    
    def _calculate_overall_complexity(self, technical: Dict, structural: Dict, code: Dict, procedural: Dict) -> int:
        """Calculate overall complexity score (1-10)."""
        # Weighted combination of complexity factors
        weights = {
            'technical': 0.4,
            'structural': 0.2,
            'code': 0.3,
            'procedural': 0.1
        }
        
        technical_score = min(10, technical['total_score'] / 10)
        structural_score = min(10, structural['structural_score'])
        code_score = min(10, code['average_complexity'])
        procedural_score = min(10, procedural['procedural_score'])
        
        overall = (
            technical_score * weights['technical'] +
            structural_score * weights['structural'] +
            code_score * weights['code'] +
            procedural_score * weights['procedural']
        )
        
        return max(1, min(10, int(overall)))
    
    def _calculate_readability(self, content: str) -> Dict[str, Any]:
        """Calculate readability metrics."""
        # Remove code blocks for readability analysis
        text_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Basic readability metrics
        sentences = len(re.findall(r'[.!?]+', text_content))
        words = len(text_content.split())
        
        # Average sentence length
        avg_sentence_length = words / max(1, sentences)
        
        # Syllable estimation (rough)
        syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in text_content.split())
        
        return {
            'sentences': sentences,
            'words': words,
            'average_sentence_length': avg_sentence_length,
            'estimated_syllables': syllables,
            'readability_score': min(10, max(1, 10 - (avg_sentence_length - 15) / 5))
        }
    
    def evaluate_for_persona(self, complexity_analysis: Dict[str, Any], persona_level: str) -> Dict[str, Any]:
        """Evaluate complexity appropriateness for a persona level."""
        level_ranges = {
            'beginner': (1, 3),
            'intermediate': (3, 6),
            'advanced': (6, 10)
        }
        
        min_score, max_score = level_ranges.get(persona_level, (3, 6))
        actual_score = complexity_analysis['overall_score']
        
        # Calculate appropriateness
        if min_score <= actual_score <= max_score:
            appropriateness = 5  # Perfect
        elif actual_score < min_score:
            appropriateness = max(1, 5 - (min_score - actual_score))
        else:
            appropriateness = max(1, 5 - (actual_score - max_score))
        
        recommendations = []
        if actual_score < min_score:
            recommendations.append(f"Content may be too simple for {persona_level} level")
            recommendations.append("Consider adding more technical depth")
        elif actual_score > max_score:
            recommendations.append(f"Content may be too complex for {persona_level} level")
            recommendations.append("Consider simplifying explanations or adding more context")
        
        return {
            'persona_level': persona_level,
            'target_range': f"{min_score}-{max_score}",
            'actual_score': actual_score,
            'appropriateness': appropriateness,
            'recommendations': recommendations
        }

def main():
    parser = argparse.ArgumentParser(description='Analyze technical complexity of documentation')
    parser.add_argument('--file', required=True, help='Path to markdown file')
    parser.add_argument('--persona-level', choices=['beginner', 'intermediate', 'advanced'], 
                       help='Evaluate for specific persona level')
    parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text')
    
    args = parser.parse_args()
    
    analyzer = ComplexityAnalyzer()
    
    with open(args.file, 'r') as f:
        content = f.read()
    
    analysis = analyzer.analyze_content(content)
    
    if args.persona_level:
        persona_eval = analyzer.evaluate_for_persona(analysis, args.persona_level)
        analysis['persona_evaluation'] = persona_eval
    
    if args.format == 'json':
        print(json.dumps(analysis, indent=2))
    elif args.format == 'yaml':
        print(yaml.dump(analysis, default_flow_style=False))
    else:
        print(f"COMPLEXITY ANALYSIS: {Path(args.file).name}")
        print(f"Overall Score: {analysis['overall_score']}/10")
        print(f"Technical Complexity: {analysis['technical_complexity']['total_score']}")
        print(f"Structural Complexity: {analysis['structural_complexity']['structural_score']}")
        print(f"Code Complexity: {analysis['code_complexity']['average_complexity']:.1f}")
        print(f"Procedural Complexity: {analysis['procedural_complexity']['procedural_score']}")
        
        if 'persona_evaluation' in analysis:
            eval_data = analysis['persona_evaluation']
            print(f"\nPERSONA EVALUATION ({eval_data['persona_level']}):")
            print(f"Target Range: {eval_data['target_range']}")
            print(f"Appropriateness: {eval_data['appropriateness']}/5")
            
            if eval_data['recommendations']:
                print("Recommendations:")
                for rec in eval_data['recommendations']:
                    print(f"  → {rec}")

if __name__ == '__main__':
    main()