# Security Policy

## ⚠️ Important — Academic Project Notice

TrustGuard AI is a **final-year academic prototype** built on synthetic data. It is **not intended for use in any production financial system**. No real customer data, live transaction streams, or regulated financial infrastructure are involved.

---

## 🔑 API Key Safety

This project uses an OpenAI API key for the RAG regulatory report feature. To keep your key safe:

- **Never commit `.streamlit/secrets.toml`** — it is listed in `.gitignore` for this reason
- Use `secrets.example.toml` as your template; replace the placeholder value locally only
- For Streamlit Cloud deployments, use the **App Settings → Secrets** panel — never hardcode keys in source files
- If you accidentally expose a key, rotate it immediately at [platform.openai.com](https://platform.openai.com/api-keys)

---

## 📦 Dependency Vulnerabilities

All dependencies are pinned in `requirements.txt`. If you discover a known CVE in a pinned dependency:

1. Check whether the vulnerability is exploitable in the context of this project
2. Open an Issue or contact the team directly (see below)
3. We will update the affected pin in `requirements.txt`

---

## 🐛 Reporting a Vulnerability

Since this is an academic project with no production deployment, there is no formal bug bounty or SLA. However, we take reports seriously.

**To report a security concern:**

1. **Do not** open a public GitHub Issue for security vulnerabilities
2. Email the project team directly — contact details are in the [About page](https://trustguard-ai-fraud-detection-c7um3xntqvxthahgld5ucm.streamlit.app/) of the live dashboard
3. Include a description of the issue, steps to reproduce, and potential impact
4. We will acknowledge the report within 5 business days

---

## 🚫 Out of Scope

The following are not considered security vulnerabilities for this project:

- Issues requiring physical access to a machine
- Social engineering attacks
- Vulnerabilities in the PaySim synthetic dataset itself
- Performance issues unrelated to security

---

## 📄 Supported Versions

| Version | Supported |
|:---|:---:|
| Current `main` branch | ✅ |
| Older branches / forks | ❌ |
