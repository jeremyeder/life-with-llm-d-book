# Consistency Check Template v1

Use this template for validating content consistency across chapters.

## Prompt Template

```
Review the following content for consistency with existing chapters. Check for:

1. **Technical Terminology**
   - Model names follow exact format: llama-3.1-8b, llama-3.1-70b, mistral-7b
   - No abbreviations: avoid llama-8b or llama-70b
   - Kubernetes terms match established usage
   - Resource specifications align with shared templates

2. **Voice and Style**
   - Technical depth matches target audience level
   - Formality consistent with existing chapters
   - Explanation patterns follow established approach
   - Code comment style matches project standards

3. **Cross-References**
   - Links to related chapters use consistent format
   - References to shared configurations are accurate
   - No duplicate content from other chapters
   - Proper attribution for shared concepts

4. **Technical Accuracy**
   - Namespace usage follows conventions: production, staging, development, llm-d-system
   - Resource specifications match docs/appendix/shared-config.md
   - Commands and configurations are accurate
   - Code examples follow project patterns

5. **Content Structure**
   - Section organization matches chapter template
   - Headers follow consistent hierarchy
   - Code blocks use proper formatting
   - Lists and bullets follow style guide

Provide specific feedback on:
- Any inconsistencies found
- Suggested corrections
- Areas that need cross-reference updates
- Technical accuracy concerns

Template Variables:
- [content] = The content to be reviewed
- [target-chapter] = Specific chapter this content belongs to
- [related-chapters] = Chapters with related content to check against
```

## Review Checklist

- [ ] Technical terms match glossary
- [ ] Voice consistent with target audience
- [ ] Cross-references are valid and helpful
- [ ] No content duplication
- [ ] Resource specs follow templates
- [ ] Code examples are tested and accurate
- [ ] Namespace usage is standard
- [ ] Structure follows chapter template

## Usage Notes

- Run this check before submitting content for human review
- Compare against 2-3 most related existing chapters
- Focus on actionable feedback for consistency improvements