import { VerificationResponse, DemoScenario } from '../types/verification';

const API_BASE = '/api';

export async function verifyText(text: string): Promise<VerificationResponse> {
  const res = await fetch(`${API_BASE}/verify/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Verification failed' }));
    throw new Error(err.detail || 'Failed to verify text');
  }
  return res.json();
}

export async function verifyUrl(url: string): Promise<VerificationResponse> {
  const res = await fetch(`${API_BASE}/verify/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'URL Verification failed' }));
    throw new Error(err.detail || 'Failed to verify URL');
  }
  return res.json();
}

export async function verifyClaim(claim: string): Promise<VerificationResponse> {
  const res = await fetch(`${API_BASE}/verify/claim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Claim verification failed' }));
    throw new Error(err.detail || 'Failed to verify claim');
  }
  return res.json();
}

export async function getDemoSamples(): Promise<DemoScenario[]> {
  const res = await fetch(`${API_BASE}/demo-samples`);
  if (!res.ok) throw new Error('Failed to load demo samples');
  return res.json();
}

export async function getAnalytics(): Promise<any> {
  const res = await fetch(`${API_BASE}/analytics`);
  if (!res.ok) throw new Error('Failed to load analytics');
  return res.json();
}
