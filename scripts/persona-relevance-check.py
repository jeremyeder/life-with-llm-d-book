#!/usr/bin/env python3
"""
Persona Relevance Analysis Tool
Analyzes chapter content against persona needs and generates validation reports.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from collections import defaultdict

class PersonaValidator:
    """Validates chapter content against persona requirements."""
    
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.personas_dir = script_dir / "personas"
        
    def load_persona(self, persona_name: str) -> Dict[str, Any]:
        """Load persona configuration from YAML file."""
        persona_file = self.personas_dir / f"{persona_name}.yaml"
        if not persona_file.exists():
            raise FileNotFoundError(f"Persona file not found: {persona_file}")
        
        with open(persona_file, 'r') as f:
            return yaml.safe_load(f)
    
    def analyze_chapter_content(self, chapter_file: Path) -> Dict[str, Any]:
        """Analyze chapter content structure and characteristics."""
        with open(chapter_file, 'r') as f:
            content = f.read()
        
        # Extract key content characteristics
        analysis = {
            'file_path': str(chapter_file),
            'chapter_name': chapter_file.stem,
            'word_count': len(content.split()),
            'sections': self._extract_sections(content),
            'code_blocks': self._extract_code_blocks(content),
            'technical_terms': self._extract_technical_terms(content),
            'practical_examples': self._count_practical_examples(content),
            'quick_start_indicators': self._detect_quick_start_content(content),
            'advanced_config_indicators': self._detect_advanced_config(content),
            'troubleshooting_content': self._detect_troubleshooting_content(content),
            'monitoring_content': self._detect_monitoring_content(content),
            'cost_content': self._detect_cost_content(content),
            'business_context': self._detect_business_context(content)
        }
        
        return analysis
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headings from markdown content."""
        sections = []
        for line in content.split('\n'):
            if line.startswith('#'):
                # Remove markdown heading syntax and clean up
                section = re.sub(r'^#+\s*', '', line).strip()
                if section:
                    sections.append(section)
        return sections
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks with language and content."""
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for language, code in matches:
            code_blocks.append({
                'language': language or 'unknown',
                'content': code.strip(),
                'lines': len(code.strip().split('\n'))
            })
        
        return code_blocks
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms and technologies mentioned."""
        # Common technical terms relevant to llm-d
        technical_patterns = [
            r'kubernetes?', r'openshift', r'llm-d', r'vllm', r'nvidia', r'gpu',
            r'inference', r'deployment', r'monitoring', r'prometheus', r'grafana',
            r'yaml', r'kubectl', r'helm', r'docker', r'container', r'pod',
            r'service', r'ingress', r'scaling', r'autoscaling', r'hpa',
            r'resource', r'cpu', r'memory', r'storage', r'pvc', r'namespace',
            r'rbac', r'security', r'tls', r'ssl', r'authentication', r'authorization'
        ]
        
        found_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_terms.extend(matches)
        
        return list(set(found_terms))
    
    def _count_practical_examples(self, content: str) -> int:
        """Count practical examples and step-by-step procedures."""
        example_indicators = [
            r'example', r'step \d+', r'procedure', r'tutorial', r'walkthrough',
            r'how to', r'let\'s', r'first,', r'next,', r'then,', r'finally,'
        ]
        
        count = 0
        for indicator in example_indicators:
            matches = re.findall(indicator, content, re.IGNORECASE)
            count += len(matches)
        
        return count
    
    def _detect_quick_start_content(self, content: str) -> bool:
        """Detect if content contains quick start or getting started material."""
        quick_start_patterns = [
            r'quick start', r'getting started', r'installation', r'setup',
            r'prerequisites', r'requirements', r'first steps', r'initial'
        ]
        
        for pattern in quick_start_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_advanced_config(self, content: str) -> bool:
        """Detect advanced configuration content."""
        advanced_patterns = [
            r'advanced', r'configuration', r'tuning', r'optimization',
            r'performance', r'scaling', r'production', r'enterprise'
        ]
        
        count = 0
        for pattern in advanced_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                count += 1
        
        return count >= 2  # Multiple advanced indicators
    
    def _detect_troubleshooting_content(self, content: str) -> bool:
        """Detect troubleshooting and debugging content."""
        troubleshooting_patterns = [
            r'troubleshooting', r'debugging', r'error', r'issue', r'problem',
            r'solution', r'fix', r'resolve', r'common problems', r'known issues'
        ]
        
        count = 0
        for pattern in troubleshooting_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            count += len(matches)
        
        return count >= 3  # Multiple troubleshooting indicators
    
    def _detect_monitoring_content(self, content: str) -> bool:
        """Detect monitoring and observability content."""
        monitoring_patterns = [
            r'monitoring', r'metrics', r'alerts', r'observability',
            r'prometheus', r'grafana', r'logging', r'traces'
        ]
        
        for pattern in monitoring_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_cost_content(self, content: str) -> bool:
        """Detect cost optimization and resource management content."""
        cost_patterns = [
            r'cost', r'pricing', r'budget', r'resource', r'optimization',
            r'efficiency', r'scaling', r'autoscaling'
        ]
        
        count = 0
        for pattern in cost_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                count += 1
        
        return count >= 2  # Multiple cost indicators
    
    def _detect_business_context(self, content: str) -> bool:
        """Detect business context and justification content."""
        business_patterns = [
            r'business', r'roi', r'value', r'benefit', r'advantage',
            r'comparison', r'alternative', r'decision', r'evaluation'
        ]
        
        for pattern in business_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def calculate_relevance_score(self, chapter_analysis: Dict[str, Any], persona: Dict[str, Any]) -> Tuple[int, List[str]]:
        """Calculate relevance score (1-5) for persona against chapter content."""
        score = 0
        reasons = []
        
        chapter_name = chapter_analysis['chapter_name']
        
        # Get persona's priority for this chapter
        chapter_priority = persona.get('chapter_priorities', {}).get(chapter_name, 3)
        
        # Base score from persona priority
        score = chapter_priority
        
        # Adjust based on content analysis
        if persona['role'] == 'Early Career Platform Engineer':
            # Alex needs quick start, clear procedures, business context
            if chapter_analysis['quick_start_indicators']:
                score = min(5, score + 1)
                reasons.append("Contains quick start content")
            
            if chapter_analysis['business_context']:
                score = min(5, score + 1)
                reasons.append("Includes business context")
            
            if chapter_analysis['practical_examples'] >= 3:
                score = min(5, score + 1)
                reasons.append("Rich with practical examples")
            
            if chapter_analysis['advanced_config_indicators']:
                score = max(1, score - 1)
                reasons.append("May be too advanced for evaluation phase")
        
        elif persona['role'] == 'Mid-Career ML Engineer':
            # Morgan needs advanced config, monitoring, troubleshooting
            if chapter_analysis['advanced_config_indicators']:
                score = min(5, score + 1)
                reasons.append("Contains advanced configuration")
            
            if chapter_analysis['monitoring_content']:
                score = min(5, score + 1)
                reasons.append("Includes monitoring content")
            
            if chapter_analysis['troubleshooting_content']:
                score = min(5, score + 1)
                reasons.append("Contains troubleshooting guidance")
            
            if not chapter_analysis['technical_terms']:
                score = max(1, score - 1)
                reasons.append("Limited technical depth")
        
        return min(5, max(1, score)), reasons
    
    def calculate_complexity_score(self, chapter_analysis: Dict[str, Any], persona: Dict[str, Any]) -> Tuple[int, List[str]]:
        """Calculate complexity appropriateness score (1-5) for persona."""
        score = 3  # Default to appropriate
        reasons = []
        
        technical_complexity = len(chapter_analysis['technical_terms'])
        code_complexity = sum(block['lines'] for block in chapter_analysis['code_blocks'])
        
        if persona['role'] == 'Early Career Platform Engineer':
            # Alex prefers moderate complexity with clear explanations
            if technical_complexity > 20:
                score = 2
                reasons.append("High technical complexity may be overwhelming")
            elif technical_complexity < 5:
                score = 2
                reasons.append("May be too basic for platform engineering role")
            else:
                score = 4
                reasons.append("Appropriate technical complexity")
            
            if code_complexity > 100:
                score = min(score, 2)
                reasons.append("Complex code examples may be difficult to follow")
        
        elif persona['role'] == 'Mid-Career ML Engineer':
            # Morgan wants deep technical content
            if technical_complexity < 10:
                score = 2
                reasons.append("Insufficient technical depth")
            elif technical_complexity > 15:
                score = 5
                reasons.append("Rich technical content")
            else:
                score = 4
                reasons.append("Good technical depth")
            
            if code_complexity < 20:
                score = min(score, 3)
                reasons.append("Could benefit from more detailed code examples")
        
        return score, reasons
    
    def validate_chapter(self, chapter_file: Path, persona_name: str) -> Dict[str, Any]:
        """Validate a chapter against a persona."""
        persona = self.load_persona(persona_name)
        chapter_analysis = self.analyze_chapter_content(chapter_file)
        
        relevance_score, relevance_reasons = self.calculate_relevance_score(chapter_analysis, persona)
        complexity_score, complexity_reasons = self.calculate_complexity_score(chapter_analysis, persona)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(chapter_analysis, persona, relevance_score, complexity_score)
        
        return {
            'chapter': chapter_analysis['chapter_name'],
            'persona': {
                'name': persona['name'],
                'role': persona['role']
            },
            'scores': {
                'relevance': relevance_score,
                'complexity': complexity_score,
                'overall': round((relevance_score + complexity_score) / 2, 1)
            },
            'analysis': {
                'relevance_reasons': relevance_reasons,
                'complexity_reasons': complexity_reasons,
                'recommendations': recommendations
            },
            'content_summary': {
                'sections': len(chapter_analysis['sections']),
                'code_blocks': len(chapter_analysis['code_blocks']),
                'technical_terms': len(chapter_analysis['technical_terms']),
                'practical_examples': chapter_analysis['practical_examples']
            }
        }
    
    def _generate_recommendations(self, chapter_analysis: Dict[str, Any], persona: Dict[str, Any], relevance_score: int, complexity_score: int) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if relevance_score < 3:
            if persona['role'] == 'Early Career Platform Engineer':
                recommendations.append("Add quick start examples and step-by-step procedures")
                recommendations.append("Include business justification and ROI examples")
            else:
                recommendations.append("Expand advanced configuration options")
                recommendations.append("Add performance optimization guidance")
        
        if complexity_score < 3:
            if persona['role'] == 'Early Career Platform Engineer':
                recommendations.append("Simplify technical explanations with more context")
                recommendations.append("Add prerequisite knowledge sections")
            else:
                recommendations.append("Increase technical depth and detailed examples")
                recommendations.append("Add advanced troubleshooting scenarios")
        
        # Content-specific recommendations
        if not chapter_analysis['troubleshooting_content'] and persona['role'] == 'Mid-Career ML Engineer':
            recommendations.append("Add comprehensive troubleshooting section")
        
        if not chapter_analysis['business_context'] and persona['role'] == 'Early Career Platform Engineer':
            recommendations.append("Include business context and decision frameworks")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Validate chapter content against persona requirements')
    parser.add_argument('--chapter', required=True, help='Path to chapter markdown file')
    parser.add_argument('--persona', required=True, help='Persona name to validate against')
    parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text', help='Output format')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Initialize validator
    validator = PersonaValidator(script_dir)
    
    try:
        # Validate chapter
        result = validator.validate_chapter(Path(args.chapter), args.persona)
        
        # Output results
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        elif args.format == 'yaml':
            print(yaml.dump(result, default_flow_style=False))
        else:
            # Text format
            print(f"PERSONA VALIDATION REPORT")
            print(f"Chapter: {result['chapter']}")
            print(f"Persona: {result['persona']['name']} ({result['persona']['role']})")
            print()
            
            print(f"SCORES:")
            print(f"  Relevance: {result['scores']['relevance']}/5")
            print(f"  Complexity: {result['scores']['complexity']}/5")
            print(f"  Overall: {result['scores']['overall']}/5")
            print()
            
            if result['analysis']['relevance_reasons']:
                print("RELEVANCE ANALYSIS:")
                for reason in result['analysis']['relevance_reasons']:
                    print(f"  ✓ {reason}")
                print()
            
            if result['analysis']['complexity_reasons']:
                print("COMPLEXITY ANALYSIS:")
                for reason in result['analysis']['complexity_reasons']:
                    print(f"  ✓ {reason}")
                print()
            
            if result['analysis']['recommendations']:
                print("RECOMMENDATIONS:")
                for rec in result['analysis']['recommendations']:
                    print(f"  → {rec}")
                print()
            
            if args.verbose:
                print("CONTENT SUMMARY:")
                print(f"  Sections: {result['content_summary']['sections']}")
                print(f"  Code blocks: {result['content_summary']['code_blocks']}")
                print(f"  Technical terms: {result['content_summary']['technical_terms']}")
                print(f"  Practical examples: {result['content_summary']['practical_examples']}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()