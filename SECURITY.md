# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in SidWeave, please report it responsibly:

1. **Do not** create a public issue
2. Email security@sidweave.dev with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Security Considerations

### Local-First Design

SidWeave is designed to run locally by default:
- No external telemetry without explicit consent
- No cloud API keys required for core functionality
- Ollama integration runs entirely locally
- Database stored in user data directory

### API Security

When running the API server:
- Default binding: `127.0.0.1` (localhost only)
- Explicit configuration required for external access
- No authentication by default (add reverse proxy for production)
- Rate limiting recommended for public deployments

### Secrets Management

- Never commit API keys, tokens, or passwords
- Use `.env` file (gitignored) for local development
- Use environment variables in production
- SidWeave reads secrets from environment/config only

### Tool Execution Safety

- MCP tools require explicit registration
- Precondition/postcondition validation
- Idempotency keys prevent duplicate execution
- Approval workflow for sensitive operations
- Timeout handling prevents hanging executions

### Data Privacy

- Task history stored locally in SQLite
- No automatic upload of execution traces
- User controls data retention
- No PII collected without consent

## Dependencies

We regularly update dependencies and monitor for vulnerabilities:
- `uv lock --upgrade` for dependency updates
- GitHub Dependabot alerts monitored
- Minimal dependency footprint

## Best Practices for Users

1. Keep SidWeave updated
2. Run API server on localhost only unless secured
3. Use strong API keys for cloud providers
4. Review tool permissions before approval
5. Monitor execution logs for anomalies
6. Backup `.sidweave` data directory regularly

## Contact

Security team: security@sidweave.dev
PGP key: Available on request