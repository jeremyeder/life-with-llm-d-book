# Security Policy

## Security Advisory

This project maintains a security-first approach to dependency management while acknowledging the constraints of working with large framework dependencies like Docusaurus.

### Current Security Status

**Last Updated:** 2025-07-27  
**Security Audit Status:** ✅ All high-severity vulnerabilities resolved

### Known Security Issues

#### Moderate Severity Issues (Accepted Risk)

**webpack-dev-server vulnerabilities (17 issues)**
- **Severity:** Moderate
- **Impact:** Development environment only (not production)
- **Status:** Accepted risk - no fix available
- **Mitigation:** These vulnerabilities affect webpack-dev-server which is:
  - Only used during development (`npm start`)
  - Not included in production builds (`npm run build`)
  - A transitive dependency of Docusaurus core
  - Will be resolved when Docusaurus updates their dependencies

#### Deprecated Dependencies (Non-Security)

The following deprecated packages are transitive dependencies that cannot be directly updated:

- `inflight@1.0.6` - Memory leak issues (via Docusaurus → webpack-dev-server → rimraf → glob)
- `rimraf@3.0.2` - Versions prior to v4 no longer supported (via Docusaurus → webpack-dev-server)
- `glob@7.2.3` - Versions prior to v9 no longer supported (via Docusaurus → webpack-dev-server → rimraf)

**Mitigation:** These are development-only dependencies that do not affect production builds.

### Security Monitoring

We actively monitor for security vulnerabilities through:

- **Automated Dependency Updates:** Dependabot with auto-merge for patch updates
- **Regular Security Audits:** `npm audit` run in CI/CD pipeline
- **Production-Only Focus:** CI fails only on high/critical production vulnerabilities

### Reporting Security Issues

If you discover a security vulnerability in this project's code (not transitive dependencies), please report it by creating a private security advisory through GitHub.

### Security Contact

For security-related questions, contact: [Repository Maintainers]

---

*This security policy is reviewed monthly and updated as needed.*