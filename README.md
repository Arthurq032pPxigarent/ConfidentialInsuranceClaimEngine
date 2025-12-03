# FHE Governance & Voting — Shareholders’ Meeting (Privacy-Preserving)

A production-oriented specification and reference implementation for **shareholders’ meeting voting** that preserves ballot privacy using the **FHE (Fully Homomorphic Encryption)** workflow. It supports multi‑model vote weighting (one‑share‑one‑vote, token‑weighted, quadratic), auditable lifecycle, and a **universal KV interface** so frontends can integrate without knowing the contract ABI.

> This repository contains:
> - `FHE_Governance_Voting.sol` — Solidity contract (≥300 lines) with a universal KV store (`isAvailable`, `setData`, `getData`) plus complete governance data structures (proposals, encrypted ballots, nullifiers, aggregated results, proof blob).
> - A React front‑end (provided separately) that only uses the KV interface and stores JSON blobs at keys like `gv_proposal_*` and `gv_result_*`.
>
> Real cryptography: replace the ciphertext/aggregation placeholders with **Zama FHE** client encryption + on‑chain verification or ZK proofs.

---

## 1) Problem Statement (User Pain Points)

- **Privacy exposure** — On‑chain votes are typically public and can bias decisions or enable coercion/bribery.  
- **Uneven weights** — Shareholders have different ownership levels; calculating trusted, auditable weights is non‑trivial.  
- **Auditability** — Traditional online voting lacks transparent, tamper‑proof audit trails.  
- **Compliance** — Corporate governance needs provable integrity and immutable records.

---

## 2) Solution (FHE‑Based)

- **Private Ballots** — Each voter submits an **encrypted ballot**; no plaintext choice is ever stored.  
- **Encrypted Aggregation** — Only **final totals** and a **proof blob** are published after the deadline.  
- **Weighting Models** — One‑share‑one‑vote, **token‑weighted**, and **quadratic** voting supported.  
- **Compliance Storage** — The full lifecycle is on‑chain, generating **immutable audit logs**.

> The reference contract is structured for an FHE workflow: ciphertexts on‑chain, homomorphic tallying off‑chain, results + proofs on‑chain.

---

## 3) Architecture

### On‑chain (Solidity)
- **KV Interface (Frontend Contract Spec)**  
  - `isAvailable()` → health check  
  - `setData(key, bytes)` → write arbitrary bytes under `key`  
  - `getData(key)` → read bytes from `key`  
  The React app uses JSON (UTF‑8) payloads under keys:
  - `gv_proposal_keys` — JSON array of ids  
  - `gv_proposal_<id>` — JSON of proposal metadata  
  - `gv_result_<id>` — JSON of aggregated totals + proof

- **Governance Model**  
  - `createProposal()` → define title, description, options, deadline, weight model  
  - `submitEncryptedBallot()` → upload ciphertext + **nullifier** (prevents duplicates)  
  - `publishAggregatedResult()` → store totals (for/against/abstain), `totalWeight`, and `proof`  
  - Optional `weightToken` (ERC‑20) and `registeredShares` mapping for weights

### Off‑chain (FHE & Services)
- **Encryption** — Zama FHE client encrypts each voter’s choice; ciphertext posted via `submitEncryptedBallot` or stored by the frontend inside the KV (`setData`).  
- **Aggregation** — Off‑chain service tallies ciphertexts homomorphically and generates a proof.  
- **Result Publication** — Organizer posts `publishAggregatedResult` and mirrors JSON into KV for the frontend to display.

---

## 4) Frontend (React) — Integration Notes

- Use only `getContractReadOnly()` and `getContractWithSigner()` helpers (no manual `new ethers.Contract`).  
- Health check via `isAvailable()` (no writes).  
- Writes go through `setData(key, bytes)`; reads use `getData(key)` and decode bytes → UTF‑8 JSON.  
- Recommended keys (already used in the sample UI):
  - `gv_proposal_keys`
  - `gv_proposal_<id>`
  - `gv_result_<id>`
  - `gv_vote_<id>_<address-lc>` (optional per‑voter record for “has voted” indicator)

---

## 5) Install & Build

### Prerequisites
- Node.js 18+  
- pnpm / npm / yarn  
- Solidity ^0.8.24 toolchain (Hardhat/Foundry)

### Quick Start (Hardhat)
1. Add the contract to your project:
   - `contracts/FHE_Governance_Voting.sol`
2. Compile:
   - `npx hardhat compile`
3. Deploy (example):
   ```ts
   // scripts/deploy.ts
   import { ethers } from "hardhat";
   async function main() {
     const Factory = await ethers.getContractFactory("FHE_Governance_Voting");
     const tokenForWeights = "0x0000000000000000000000000000000000000000"; // or ERC20 address
     const inst = await Factory.deploy(ethers.ZeroAddress, tokenForWeights);
     await inst.waitForDeployment();
     console.log("FHE_Governance_Voting:", await inst.getAddress());
   }
   main().catch((e) => { console.error(e); process.exit(1); });
   ```
4. Frontend:
   - Use your existing `getContractReadOnly()` / `getContractWithSigner()` wired to the deployed address/ABI.

---

## 6) Usage Flow

1. Organizer **creates a proposal** (`createProposal`).  
2. Shareholders **submit encrypted ballots** (`submitEncryptedBallot`) with proper nullifiers.  
3. After deadline, organizer posts **aggregated totals + proof** (`publishAggregatedResult`).  
4. An off‑chain service (or the organizer) **mirrors JSON blobs into KV** (`setData`) so the generic UI can render lists and results.  
5. Auditors verify the **proof** and the on‑chain event log.

---

## 7) Security & Compliance

- **No plaintext ballots on‑chain**; only ciphertext and totals.  
- **Nullifier** prevents duplicate voting per proposal and address.  
- **Immutable logs** — Every step emits events for audits.  
- **Pluggable weighting** — Use ERC‑20 balances or registered shares.  
- **Replaceable crypto** — Swap the placeholder with real FHE client + on‑chain verification when available.

---

## 8) Future Enhancements

- On‑chain FHE helpers / precompiles (when supported).  
- ZK proofs to attest that aggregation matches ciphertext set.  
- Snapshot‑based token weighting at block height `N`.  
- Off‑chain allowlists (soulbound, KYC‑based) with privacy‑preserving checks.  
- Multi‑proposal batched finalization & IPFS/Arweave anchoring for artifacts.

---

**License:** MIT  
**Author:** FHE Governance (2025)
