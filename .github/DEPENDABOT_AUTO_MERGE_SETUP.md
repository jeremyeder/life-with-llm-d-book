# Dependabot Auto-merge Setup Guide

This repository is configured for automatic dependency updates with the following features:

## 🤖 What's Automated

### Automatic Merging
- **Patch updates** (1.0.0 → 1.0.1) - Auto-merged immediately
- **Minor updates** (1.0.0 → 1.1.0) - Auto-merged immediately  
- **GitHub Actions** - Auto-merged immediately
- **Major updates** (1.0.0 → 2.0.0) - Requires manual review

### Grouping
- **Docusaurus packages** - Grouped into single PR
- **Dev dependencies** - Grouped into single PR
- **GitHub Actions** - Grouped into single PR

## 📁 Configuration Files

### `.github/dependabot.yml`
- Schedules weekly updates (Mondays at 9 AM)
- Groups related dependencies
- Limits open PRs to prevent spam

### `.github/workflows/dependabot-auto-merge.yml`
- Auto-approves and merges safe updates
- Comments on major updates for manual review
- Runs security checks before merging

## ⚙️ Repository Settings

To enable auto-merge, ensure these repository settings:

### 1. General Settings
- ✅ Allow auto-merge
- ✅ Allow merge commits
- ✅ Automatically delete head branches

### 2. Branch Protection (Optional but Recommended)
```
Branch: main
✅ Require a pull request before merging
✅ Require status checks to pass before merging
   - Test Examples with Coverage
✅ Restrict pushes that create files
✅ Allow force pushes: Specify who can force push (yourself only)
```

### 3. Actions Permissions
- ✅ Allow GitHub Actions to create and approve pull requests

## 🔧 Manual Configuration Steps

1. **Enable auto-merge in repository settings:**
   ```
   Settings → General → Pull Requests → Allow auto-merge
   ```

2. **Set up branch protection (recommended):**
   ```bash
   gh api repos/jeremyeder/life-with-llm-d-book/branches/main/protection \
     --method PUT \
     --field required_status_checks='{"strict":true,"checks":[{"context":"Test Examples with Coverage"}]}' \
     --field enforce_admins=false \
     --field required_pull_request_reviews='{"required_approving_review_count":0,"dismiss_stale_reviews":false}' \
     --field restrictions=null
   ```

3. **Test the setup:**
   - Create a test dependency update
   - Verify auto-merge workflow runs
   - Check that tests pass before merging

## 🛡️ Security Considerations

- Only patch/minor updates are auto-merged
- All updates still run through CI tests
- Major version updates require manual review
- Security advisories trigger immediate notifications

## 📊 Monitoring

Monitor auto-merge activity in:
- Actions tab → Dependabot Auto-merge workflow
- Pull requests → Closed (merged by dependabot)
- Security tab → Dependabot alerts

## 🔄 Rollback Plan

If auto-merge causes issues:
1. Disable the workflow: Comment out jobs in `dependabot-auto-merge.yml`
2. Revert problematic updates: `git revert <commit-hash>`
3. Review and fix issues manually
4. Re-enable with stricter conditions

## 📞 Troubleshooting

**Auto-merge not working?**
- Check repository settings allow auto-merge
- Verify workflow permissions in Actions
- Ensure branch protection allows the workflow to merge

**Too many PRs?**
- Reduce `open-pull-requests-limit` in dependabot.yml
- Add more dependency patterns to groups
- Change schedule to monthly instead of weekly