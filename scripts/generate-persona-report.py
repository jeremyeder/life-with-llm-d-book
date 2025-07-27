#!/usr/bin/env python3
"""
Generate comprehensive persona validation report across all chapters.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import sys

class PersonaReportGenerator:
    """Generates comprehensive persona validation reports."""
    
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.project_root = script_dir.parent
        self.docs_dir = self.project_root / "docs"
        
    def find_chapter_files(self) -> List[Path]:
        """Find all chapter markdown files."""
        chapter_files = []
        
        # Pattern for numbered chapters
        for i in range(1, 13):  # Chapters 1-12
            for pattern in [f"{i:02d}-*.md", f"{i:02d}_*.md"]:
                matches = list(self.docs_dir.glob(pattern))
                chapter_files.extend(matches)
        
        # Sort by filename
        chapter_files.sort()
        return chapter_files
    
    def run_persona_validation(self, chapter_file: Path, persona: str) -> Dict[str, Any]:
        """Run persona validation for a chapter."""
        try:
            result = subprocess.run([
                'python3', 
                str(self.script_dir / 'persona-relevance-check.py'),
                '--chapter', str(chapter_file),
                '--persona', persona,
                '--format', 'json'
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            return {
                'error': f"Validation failed: {e.stderr}",
                'chapter': chapter_file.stem,
                'persona': persona
            }
    
    def generate_comprehensive_report(self, personas: List[str]) -> Dict[str, Any]:
        """Generate comprehensive report for all personas and chapters."""
        chapters = self.find_chapter_files()
        
        report = {
            'summary': {
                'total_chapters': len(chapters),
                'personas_evaluated': len(personas),
                'generation_timestamp': subprocess.check_output(['date'], text=True).strip()
            },
            'persona_reports': {},
            'chapter_analysis': {},
            'recommendations': {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            }
        }
        
        # Generate persona-specific reports
        for persona in personas:
            print(f"Generating report for {persona}...", file=sys.stderr)
            persona_data = {
                'persona': persona,
                'chapters': {},
                'summary': {
                    'total_score': 0,
                    'relevance_score': 0,
                    'complexity_score': 0,
                    'chapter_count': 0
                }
            }
            
            for chapter in chapters:
                print(f"  Validating {chapter.name}...", file=sys.stderr)
                validation_result = self.run_persona_validation(chapter, persona)
                
                if 'error' not in validation_result:
                    persona_data['chapters'][chapter.stem] = validation_result
                    
                    # Update summary
                    scores = validation_result.get('scores', {})
                    persona_data['summary']['total_score'] += scores.get('overall', 0)
                    persona_data['summary']['relevance_score'] += scores.get('relevance', 0)
                    persona_data['summary']['complexity_score'] += scores.get('complexity', 0)
                    persona_data['summary']['chapter_count'] += 1
                else:
                    persona_data['chapters'][chapter.stem] = validation_result
            
            # Calculate averages
            if persona_data['summary']['chapter_count'] > 0:
                count = persona_data['summary']['chapter_count']
                persona_data['summary']['average_total'] = persona_data['summary']['total_score'] / count
                persona_data['summary']['average_relevance'] = persona_data['summary']['relevance_score'] / count
                persona_data['summary']['average_complexity'] = persona_data['summary']['complexity_score'] / count
            
            report['persona_reports'][persona] = persona_data
        
        # Generate chapter analysis
        for chapter in chapters:
            chapter_data = {
                'chapter': chapter.stem,
                'persona_scores': {},
                'recommendations': []
            }
            
            for persona in personas:
                if chapter.stem in report['persona_reports'][persona]['chapters']:
                    chapter_validation = report['persona_reports'][persona]['chapters'][chapter.stem]
                    if 'scores' in chapter_validation:
                        chapter_data['persona_scores'][persona] = chapter_validation['scores']
                        
                        # Collect recommendations
                        recs = chapter_validation.get('analysis', {}).get('recommendations', [])
                        for rec in recs:
                            chapter_data['recommendations'].append(f"[{persona}] {rec}")
            
            report['chapter_analysis'][chapter.stem] = chapter_data
        
        # Generate prioritized recommendations
        self._generate_prioritized_recommendations(report)
        
        return report
    
    def _generate_prioritized_recommendations(self, report: Dict[str, Any]) -> None:
        """Generate prioritized recommendations based on analysis."""
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for persona, persona_data in report['persona_reports'].items():
            avg_scores = persona_data['summary']
            
            # High priority: Low overall scores
            if avg_scores.get('average_total', 0) < 3:
                high_priority.append(f"Overall content relevance too low for {persona} (avg: {avg_scores.get('average_total', 0):.1f})")
            
            # Medium priority: Complexity mismatches
            if avg_scores.get('average_complexity', 0) < 2.5:
                medium_priority.append(f"Content complexity too low for {persona}")
            elif avg_scores.get('average_complexity', 0) > 4:
                medium_priority.append(f"Content complexity too high for {persona}")
            
            # Chapter-specific recommendations
            for chapter, chapter_data in persona_data['chapters'].items():
                if 'scores' in chapter_data:
                    scores = chapter_data['scores']
                    
                    # High priority: Very low scores
                    if scores.get('overall', 0) < 2:
                        high_priority.append(f"Chapter {chapter} needs major revision for {persona}")
                    
                    # Medium priority: Moderate issues
                    elif scores.get('overall', 0) < 3:
                        medium_priority.append(f"Chapter {chapter} needs improvement for {persona}")
                    
                    # Low priority: Minor improvements
                    elif scores.get('overall', 0) < 4:
                        low_priority.append(f"Chapter {chapter} could be enhanced for {persona}")
        
        report['recommendations']['high_priority'] = list(set(high_priority))
        report['recommendations']['medium_priority'] = list(set(medium_priority))
        report['recommendations']['low_priority'] = list(set(low_priority))
    
    def format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as readable text."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("PERSONA VALIDATION COMPREHENSIVE REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report['summary']['generation_timestamp']}")
        lines.append(f"Chapters analyzed: {report['summary']['total_chapters']}")
        lines.append(f"Personas evaluated: {report['summary']['personas_evaluated']}")
        lines.append("")
        
        # Persona summaries
        lines.append("PERSONA SUMMARIES")
        lines.append("-" * 20)
        for persona, data in report['persona_reports'].items():
            summary = data['summary']
            lines.append(f"\n{persona}:")
            lines.append(f"  Average Overall Score: {summary.get('average_total', 0):.1f}/5")
            lines.append(f"  Average Relevance: {summary.get('average_relevance', 0):.1f}/5")
            lines.append(f"  Average Complexity: {summary.get('average_complexity', 0):.1f}/5")
            lines.append(f"  Chapters evaluated: {summary.get('chapter_count', 0)}")
        
        # Priority recommendations
        lines.append("\n\nPRIORITIZED RECOMMENDATIONS")
        lines.append("-" * 30)
        
        if report['recommendations']['high_priority']:
            lines.append("\nHIGH PRIORITY:")
            for rec in report['recommendations']['high_priority']:
                lines.append(f"  🔴 {rec}")
        
        if report['recommendations']['medium_priority']:
            lines.append("\nMEDIUM PRIORITY:")
            for rec in report['recommendations']['medium_priority']:
                lines.append(f"  🟡 {rec}")
        
        if report['recommendations']['low_priority']:
            lines.append("\nLOW PRIORITY:")
            for rec in report['recommendations']['low_priority']:
                lines.append(f"  🟢 {rec}")
        
        # Chapter-by-chapter analysis
        lines.append("\n\nCHAPTER-BY-CHAPTER ANALYSIS")
        lines.append("-" * 35)
        
        for chapter, data in report['chapter_analysis'].items():
            lines.append(f"\n{chapter}:")
            
            # Persona scores
            for persona, scores in data['persona_scores'].items():
                lines.append(f"  {persona}: {scores.get('overall', 0):.1f}/5 (R:{scores.get('relevance', 0)}/5, C:{scores.get('complexity', 0)}/5)")
            
            # Recommendations for this chapter
            if data['recommendations']:
                lines.append("  Recommendations:")
                for rec in data['recommendations'][:3]:  # Show top 3
                    lines.append(f"    → {rec}")
        
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive persona validation report')
    parser.add_argument('--personas', nargs='+', 
                       default=['alex-platform-engineer', 'morgan-ml-engineer'],
                       help='Personas to include in report')
    parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text',
                       help='Output format')
    parser.add_argument('--output', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Generate report
    generator = PersonaReportGenerator(script_dir)
    report = generator.generate_comprehensive_report(args.personas)
    
    # Format output
    if args.format == 'json':
        output = json.dumps(report, indent=2)
    elif args.format == 'yaml':
        output = yaml.dump(report, default_flow_style=False)
    else:
        output = generator.format_text_report(report)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

if __name__ == '__main__':
    main()