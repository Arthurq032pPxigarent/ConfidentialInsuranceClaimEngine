// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/*
    Confidential Insurance Claims Engine (FHE-enabled)
    -------------------------------------------------
    Purpose:
      - Enable multiparty encrypted input collection, joint modeling, encrypted training/inference,
        differential-privacy-aware access control, and time-sliced encrypted aggregation + billing.
      - Built as a template integrating Zama-style FHE primitives (imported as `FHE`) with on-chain
        access-control and auditability. All sensitive values remain encrypted on-chain; decryption
        occurs only through the FHE oracle callbacks / proof-verified procedures.

    Notes:
      - This contract is a high-level, production-oriented template. Integrators must adapt names
        of FHE calls (e.g. requestComputation / requestDecryption / checkSignatures) to the actual
        FHE Solidity library API available in their environment.
      - Comments and function descriptions are in English as requested.
*/

import { FHE, euint32, ebool } from "@fhevm/solidity/lib/FHE.sol";
import { SepoliaConfig } from "@fhevm/solidity/config/ZamaConfig.sol";

contract ConfidentialInsuranceClaimsFHE is SepoliaConfig {
    // -----------------------
    // Roles & Access Control
    // -----------------------
    address public owner;

    mapping(address => bool) public admins;
    mapping(address => bool) public insurers;
    mapping(address => bool) public auditors;
    mapping(address => bool) public dataProviders; // parties that supply encrypted inputs

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyAdmin() {
        require(admins[msg.sender], "Not admin");
        _;
    }

    modifier onlyInsurer() {
        require(insurers[msg.sender], "Not insurer");
        _;
    }

    modifier onlyAuditor() {
        require(auditors[msg.sender], "Not auditor");
        _;
    }

    // -----------------------
    // Basic types & structs
    // -----------------------

    // Encrypted claim submitted by a policyholder or data provider.
    struct EncryptedClaim {
        uint256 claimId;
        address submitter;
        euint32 encryptedPolicyFeatures; // encrypted numeric vector (packed / or pointer)
        euint32 encryptedSensorSnapshot; // encrypted sensor / meter reading snapshot
        euint32 encryptedAux;            // encrypted aux data (e.g. geolocation hash, timestamp hash)
        uint256 submittedAt;
        bool processed;                  // whether claim was already consumed for model or billing
    }

    // Metadata for an encrypted model artifact (weights, versioning)
    struct EncryptedModel {
        uint256 modelId;
        address owner;               // administrative owner of the model (could be multisig)
        bytes32[] encryptedWeights;  // ciphertexts representing encrypted model parameters
        bytes32 modelMetadataHash;   // hash of metadata (schema, hyperparams) stored off-chain
        uint256 createdAt;
        uint256 trainingRequestId;   // FHE request id for last training job (if any)
        bool isActive;
    }

    // Differential privacy / access control settings that influence decryption requests
    struct DPParams {
        // per-request epsilon (as encrypted or small plaintext param)
        // NOTE: For simplicity we store a plaintext policy threshold here; in practice this may be managed off-chain.
        uint32 maxEpsilonPermitted; 
        bool requireAuditorApproval; // whether decryption requests require auditor signature
    }

    // Time-slice aggregation record for meter / energy usage / interval data
    struct EncryptedTimesliceAggregate {
        bytes32 timesliceId;     // id (e.g., keccak(timestamp block / interval))
        euint32 encryptedSum;    // encrypted aggregated usage for that interval
        uint256 startTs;
        uint256 endTs;
        bool billed;
    }

    // Billing record referencing encrypted aggregate -> encrypted cost
    struct EncryptedBilling {
        uint256 billId;
        address consumer;        // who is billed (on-chain identity)
        bytes32 timesliceId;
        euint32 encryptedCost;   // encrypted cost (units * price) kept encrypted on-chain
        uint256 createdAt;
        bool paid;               // paid flag (payment settlement may occur on-chain or off-chain)
    }

    // -----------------------
    // Contract storage
    // -----------------------
    uint256 public nextClaimId;
    uint256 public nextModelId;
    uint256 public nextBillId;

    mapping(uint256 => EncryptedClaim) public claims;
    mapping(uint256 => EncryptedModel) public models;

    // timesliceId (bytes32) => aggregate
    mapping(bytes32 => EncryptedTimesliceAggregate) public timesliceAggregates;

    // billId => billing
    mapping(uint256 => EncryptedBilling) public bills;

    DPParams public dpPolicy;

    // Mapping from FHE request id -> context for callbacks
    enum RequestKind { NONE, DECRYPTION, TRAINING, INFERENCE, AGGREGATION }
    struct RequestContext {
        RequestKind kind;
        uint256 subjectId;      // claimId, modelId or billId depending on kind
        address requester;      // who asked the request
        bytes32 meta;           // optional metadata (e.g., timesliceId encoded)
    }
    mapping(uint256 => RequestContext) public fheRequestContexts;

    // Events for observability
    event ClaimSubmitted(uint256 indexed claimId, address indexed submitter, uint256 timestamp);
    event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 timestamp);
    event TrainingRequested(uint256 indexed modelId, uint256 requestId);
    event InferenceRequested(uint256 indexed modelId, uint256 claimId, uint256 requestId);
    event DecryptionRequested(uint256 indexed subjectId, uint256 requestId, RequestKind kind);
    event DecryptionCompleted(uint256 indexed requestId, uint256 subjectId);
    event TimesliceAggregated(bytes32 indexed timesliceId, uint256 requestId);
    event BillingCreated(uint256 indexed billId, address indexed consumer, bytes32 timesliceId);
    event RoleUpdated(address indexed who, string role, bool enabled);

    // -----------------------
    // Constructor & role mgmt
    // -----------------------
    constructor() {
        owner = msg.sender;
        admins[msg.sender] = true;
        insurers[msg.sender] = true;
        auditors[msg.sender] = false;
        dpPolicy = DPParams({ maxEpsilonPermitted: 1000, requireAuditorApproval: true }); // default policy (placeholder)
    }

    /// @notice Add or remove an admin
    function setAdmin(address who, bool enabled) external onlyOwner {
        admins[who] = enabled;
        emit RoleUpdated(who, "admin", enabled);
    }

    /// @notice Add or remove an insurer
    function setInsurer(address who, bool enabled) external onlyOwner {
        insurers[who] = enabled;
        emit RoleUpdated(who, "insurer", enabled);
    }

    /// @notice Add or remove an auditor
    function setAuditor(address who, bool enabled) external onlyOwner {
        auditors[who] = enabled;
        emit RoleUpdated(who, "auditor", enabled);
    }

    /// @notice Add or remove a data provider
    function setDataProvider(address who, bool enabled) external onlyAdmin {
        dataProviders[who] = enabled;
        emit RoleUpdated(who, "dataProvider", enabled);
    }

    /// @notice Update on-chain DP policy (small plaintext governance params)
    function updateDPPolicy(uint32 maxEps, bool requireAuditor) external onlyAdmin {
        dpPolicy.maxEpsilonPermitted = maxEps;
        dpPolicy.requireAuditorApproval = requireAuditor;
    }

    // -----------------------
    // Claim lifecycle
    // -----------------------

    /// @notice Submit an encrypted claim. All sensitive fields are FHE ciphertexts (euint32).
    /// @dev submitter must be a data provider or the claimant themselves (access control optional).
    function submitEncryptedClaim(
        euint32 encryptedPolicyFeatures,
        euint32 encryptedSensorSnapshot,
        euint32 encryptedAux
    ) external returns (uint256) {
        // Optional: require a recognized data provider or allow public.
        // require(dataProviders[msg.sender], "Not authorized data provider");

        nextClaimId += 1;
        uint256 id = nextClaimId;

        claims[id] = EncryptedClaim({
            claimId: id,
            submitter: msg.sender,
            encryptedPolicyFeatures: encryptedPolicyFeatures,
            encryptedSensorSnapshot: encryptedSensorSnapshot,
            encryptedAux: encryptedAux,
            submittedAt: block.timestamp,
            processed: false
        });

        emit ClaimSubmitted(id, msg.sender, block.timestamp);
        return id;
    }

    // -----------------------
    // Model registration & lifecycle
    // -----------------------

    /// @notice Register an encrypted model artifact (e.g., initial weights encrypted off-chain)
    function registerEncryptedModel(bytes32[] calldata encryptedWeights, bytes32 metadataHash) external returns (uint256) {
        nextModelId += 1;
        uint256 id = nextModelId;

        EncryptedModel storage m = models[id];
        m.modelId = id;
        m.owner = msg.sender;
        m.encryptedWeights = encryptedWeights;
        m.modelMetadataHash = metadataHash;
        m.createdAt = block.timestamp;
        m.isActive = true;

        emit ModelRegistered(id, msg.sender, block.timestamp);
        return id;
    }

    /// @notice Request a training job on encrypted claims (multiparty joint modeling)
    /// @dev Aggregation of multiple encrypted claims can be done off-chain or via repeated FHE.add on-chain.
    ///      Here we request an FHE-side training job that takes ciphertext pointers (contract will provide list via ABI or index).
    function requestModelTraining(uint256 modelId, uint256[] calldata claimIds, bytes32 trainingMetadata) external onlyAdmin returns (uint256) {
        require(models[modelId].isActive, "Model not active");

        // Build list of ciphertexts for training: for simplicity, we convert each euint32 into bytes32 via FHE.toBytes32.
        // NOTE: If training expects tensors, the off-chain FHE runtime will look them up by the provided ciphertext set.
        bytes32[] memory ciphertexts = new bytes32[](claimIds.length * 3 + models[modelId].encryptedWeights.length);
        uint256 idx = 0;

        // include model weights as inputs (so training runs on top of existing weights)
        for (uint i = 0; i < models[modelId].encryptedWeights.length; i++) {
            ciphertexts[idx++] = models[modelId].encryptedWeights[i];
        }

        // include each claim's encrypted fields
        for (uint j = 0; j < claimIds.length; j++) {
            EncryptedClaim storage c = claims[claimIds[j]];
            ciphertexts[idx++] = FHE.toBytes32(c.encryptedPolicyFeatures);
            ciphertexts[idx++] = FHE.toBytes32(c.encryptedSensorSnapshot);
            ciphertexts[idx++] = FHE.toBytes32(c.encryptedAux);
        }

        // Request training/evaluation on FHE runtime. The runtime will call back to onTrainingComplete.
        uint256 reqId = FHE.requestComputation(ciphertexts, this.onTrainingComplete.selector);
        fheRequestContexts[reqId] = RequestContext({
            kind: RequestKind.TRAINING,
            subjectId: modelId,
            requester: msg.sender,
            meta: trainingMetadata
        });

        models[modelId].trainingRequestId = reqId;

        emit TrainingRequested(modelId, reqId);
        return reqId;
    }

    /// @notice Callback invoked by the FHE runtime once training completes. The runtime supplies
    ///         encrypted updated weights, plus an integrity proof (checked via FHE.checkSignatures).
    /// @dev cleartexts here are encrypted weights packaged by the runtime (commonly bytes representing ciphertexts),
    ///      and proof is the runtime signature/proof. We must verify the proof before accepting.
    function onTrainingComplete(uint256 requestId, bytes memory runtimePayload, bytes memory proof) public {
        RequestContext storage ctx = fheRequestContexts[requestId];
        require(ctx.kind == RequestKind.TRAINING, "Invalid request kind");

        // verify integrity of the computation via FHE library (reverts on invalid)
        FHE.checkSignatures(requestId, runtimePayload, proof);

        // decode runtimePayload -> bytes32[] updatedWeights
        bytes32[] memory updatedWeights = abi.decode(runtimePayload, (bytes32[]));

        // Persist updated encrypted weights into the model artifact
        EncryptedModel storage m = models[ctx.subjectId];
        delete m.encryptedWeights;
        for (uint i = 0; i < updatedWeights.length; i++) {
            m.encryptedWeights.push(updatedWeights[i]);
        }

        // optionally: mark all claims used as processed
        // (security: we don't mark automatically here to remain auditable - operator may mark)
        emit DecryptionCompleted(requestId, ctx.subjectId);
    }

    // -----------------------
    // Inference
    // -----------------------

    /// @notice Request an encrypted inference: apply model to a claim without revealing plaintexts.
    ///         The FHE runtime returns encrypted prediction (e.g., pay-out score) through callback.
    function requestEncryptedInference(uint256 modelId, uint256 claimId) external onlyInsurer returns (uint256) {
        require(models[modelId].isActive, "Model not active");
        EncryptedClaim storage c = claims[claimId];

        // Build ciphertext array: model weights + claim fields
        bytes32[] memory ciphertexts = new bytes32[](models[modelId].encryptedWeights.length + 3);
        uint256 idx = 0;
        for (uint i = 0; i < models[modelId].encryptedWeights.length; i++) {
            ciphertexts[idx++] = models[modelId].encryptedWeights[i];
        }
        ciphertexts[idx++] = FHE.toBytes32(c.encryptedPolicyFeatures);
        ciphertexts[idx++] = FHE.toBytes32(c.encryptedSensorSnapshot);
        ciphertexts[idx++] = FHE.toBytes32(c.encryptedAux);

        // Request the FHE runtime to run inference; callback goes to onInferenceComplete.
        uint256 reqId = FHE.requestComputation(ciphertexts, this.onInferenceComplete.selector);
        fheRequestContexts[reqId] = RequestContext({
            kind: RequestKind.INFERENCE,
            subjectId: claimId,
            requester: msg.sender,
            meta: bytes32(modelId) // pack modelId into meta for reference
        });

        emit InferenceRequested(modelId, claimId, reqId);
        return reqId;
    }

    /// @notice Callback for encrypted inference results. The runtime must sign the result.
    ///         The inference result remains encrypted; it can be stored on-chain as ciphertext for later
    ///         decryption (subject to DP and access control).
    function onInferenceComplete(uint256 requestId, bytes memory runtimePayload, bytes memory proof) public {
        RequestContext storage ctx = fheRequestContexts[requestId];
        require(ctx.kind == RequestKind.INFERENCE, "Invalid request kind");

        // Verify runtime proof
        FHE.checkSignatures(requestId, runtimePayload, proof);

        // Expect runtimePayload to be bytes32[] containing encrypted prediction ciphertexts
        bytes32[] memory encryptedResult = abi.decode(runtimePayload, (bytes32[]));

        // For simplicity, store the first ciphertext into the claim.encryptedAux to keep association.
        // In a real deployment we'd have a dedicated storage for encrypted inference results.
        EncryptedClaim storage c = claims[ctx.subjectId];
        if (encryptedResult.length > 0) {
            // Overwrite encryptedAux with returned ciphertext (keep previous aux if needed elsewhere)
            c.encryptedAux = FHE.fromBytes32(encryptedResult[0]);
        }

        emit DecryptionCompleted(requestId, ctx.subjectId);
    }

    // -----------------------
    // Differential Privacy & Controlled Decryption
    // -----------------------

    /// @notice Request decryption of a stored ciphertext (e.g. an inference score or category count).
    ///         This function applies on-chain policy checks (DP limits, auditor approval), then requests
    ///         the FHE oracle to decrypt under proof.
    function requestDecryption(uint256 subjectId, bytes32 meta, uint32 requestedEpsilon) external returns (uint256) {
        // subjectId meaning depends on caller: claimId, billId, modelId etc. Caller must ensure they have rights.

        // Enforce DP policy: requestedEpsilon must be <= allowed threshold (simple on-chain policy)
        require(requestedEpsilon <= dpPolicy.maxEpsilonPermitted, "DP epsilon too large");

        // If auditor approval required, ensure caller is an auditor or auditor has pre-approved off-chain (simplified)
        if (dpPolicy.requireAuditorApproval) {
            require(auditors[msg.sender] || admins[msg.sender], "Auditor approval required");
        }

        // For demo: pick ciphertexts to decrypt based on subjectId. Here we demonstrate claim-level decryption:
        EncryptedClaim storage c = claims[subjectId];
        require(c.claimId != 0, "Claim not found");

        bytes32;
        ciphertexts[0] = FHE.toBytes32(c.encryptedPolicyFeatures);
        ciphertexts[1] = FHE.toBytes32(c.encryptedSensorSnapshot);
        ciphertexts[2] = FHE.toBytes32(c.encryptedAux);

        uint256 reqId = FHE.requestDecryption(ciphertexts, this.onDecryptionComplete.selector);

        fheRequestContexts[reqId] = RequestContext({
            kind: RequestKind.DECRYPTION,
            subjectId: subjectId,
            requester: msg.sender,
            meta: meta
        });

        emit DecryptionRequested(subjectId, reqId, RequestKind.DECRYPTION);
        return reqId;
    }

    /// @notice Callback that receives plaintexts (cleartexts) from the FHE runtime along with proofs.
    ///         This function MUST verify the proof before taking any action.
    function onDecryptionComplete(uint256 requestId, bytes memory cleartexts, bytes memory proof) public {
        RequestContext storage ctx = fheRequestContexts[requestId];
        require(ctx.kind == RequestKind.DECRYPTION, "Invalid request kind");

        // Verify signature/proof from FHE runtime
        FHE.checkSignatures(requestId, cleartexts, proof);

        // Here we decode the cleartexts and optionally emit an event or store an audit. We avoid storing
        // plaintext on-chain long-term: prefer emitting an event or handing results to an off-chain system.
        string[] memory results = abi.decode(cleartexts, (string[]));
        // Example: results[0] = policy features, results[1] = sensor snapshot, results[2] = aux

        // Emit an event so authorized off-chain services can pick it up for settlement / UI.
        // Note: event may contain sensitive plaintext — in production you may instead transmit via encrypted
        // off-chain channels or reveal only to authorized parties.
        emit DecryptionCompleted(requestId, ctx.subjectId);
    }

    // -----------------------
    // Time-sliced aggregation & privacy-preserving billing
    // -----------------------

    /// @notice Submit or update an encrypted timeslice contribution.
    ///         Multiple providers can contribute encrypted partials which are aggregated homomorphically.
    function submitTimesliceContribution(bytes32 timesliceId, euint32 encryptedContribution) external {
        // Accept contributions from data providers or policyholders
        require(dataProviders[msg.sender] || insurers[msg.sender] || admins[msg.sender], "Not authorized");

        EncryptedTimesliceAggregate storage agg = timesliceAggregates[timesliceId];

        if (agg.startTs == 0) {
            // first contribution: initialize timeslice
            agg.timesliceId = timesliceId;
            agg.startTs = block.timestamp; // approximate; in practice caller should provide start/end
            agg.endTs = block.timestamp;
            agg.encryptedSum = encryptedContribution;
            agg.billed = false;
        } else {
            // homomorphically add contribution to the existing encrypted sum
            agg.encryptedSum = FHE.add(agg.encryptedSum, encryptedContribution);
            // update end timestamp
            agg.endTs = block.timestamp;
        }

        // We do not publish plaintext: sum remains encrypted for downstream billing / DP mechanisms.
    }

    /// @notice Request on-chain FHE aggregation finalization and billing computation.
    ///         This issues a computation request to the FHE runtime which will compute encrypted cost =
    ///         encryptedSum * (encryptedPriceOrPlainPrice) and return ciphertext(s) for storage as a bill.
    function requestTimesliceBilling(bytes32 timesliceId, uint256 consumerAccountId, euint32 encryptedPricePerUnit) external onlyInsurer returns (uint256) {
        EncryptedTimesliceAggregate storage agg = timesliceAggregates[timesliceId];
        require(agg.startTs != 0, "Timeslice not found");
        require(!agg.billed, "Already billed");

        // Prepare inputs: aggregated encryptedSum + encryptedPricePerUnit (or price in plaintext pushed as FHE.asEuint32)
        bytes32;
        ciphertexts[0] = FHE.toBytes32(agg.encryptedSum);
        ciphertexts[1] = FHE.toBytes32(encryptedPricePerUnit);

        uint256 reqId = FHE.requestComputation(ciphertexts, this.onBillingComputationComplete.selector);

        // Pack metadata: store timesliceId in meta so callback can create bill
        fheRequestContexts[reqId] = RequestContext({
            kind: RequestKind.AGGREGATION,
            subjectId: 0, // will be filled in callback via meta->timeslice
            requester: msg.sender,
            meta: timesliceId
        });

        emit TimesliceAggregated(timesliceId, reqId);
        return reqId;
    }

    /// @notice Callback once billing computation completes. The runtime returns encrypted cost ciphertext(s).
    function onBillingComputationComplete(uint256 requestId, bytes memory runtimePayload, bytes memory proof) public {
        RequestContext storage ctx = fheRequestContexts[requestId];
        require(ctx.kind == RequestKind.AGGREGATION, "Invalid request kind");

        // Verify proof
        FHE.checkSignatures(requestId, runtimePayload, proof);

        // Extract encrypted cost ciphertexts
        bytes32[] memory encryptedCosts = abi.decode(runtimePayload, (bytes32[]));
        require(encryptedCosts.length > 0, "No encrypted cost returned");

        // create a billing record with the returned encrypted cost (store as euint32)
        nextBillId += 1;
        uint256 billId = nextBillId;

        EncryptedBilling storage b = bills[billId];
        b.billId = billId;
        // Note: for demo we don't link consumer to an on-chain account mapping; consumer info is off-chain ID
        b.consumer = ctx.requester; // insurer requested billing on behalf of consumer; replace as needed
        b.timesliceId = ctx.meta;
        b.encryptedCost = FHE.fromBytes32(encryptedCosts[0]);
        b.createdAt = block.timestamp;
        b.paid = false;

        // mark timeslice billed
        bytes32 timesliceId = ctx.meta;
        EncryptedTimesliceAggregate storage agg = timesliceAggregates[timesliceId];
        agg.billed = true;

        emit BillingCreated(billId, b.consumer, timesliceId);
    }

    // -----------------------
    // Utilities
    // -----------------------

    /// @notice Helper to convert bytes32[] stored as model.weights into viewable length
    function getModelWeightsLength(uint256 modelId) external view returns (uint256) {
        return models[modelId].encryptedWeights.length;
    }

    /// @notice Expose an encrypted timeslice aggregate ciphertext for off-chain consumers
    function getTimesliceEncryptedSum(bytes32 timesliceId) external view returns (euint32) {
        return timesliceAggregates[timesliceId].encryptedSum;
    }

    /// @notice Fetch a billing record's encrypted cost ciphertext (to be decrypted by authorized parties)
    function getEncryptedBillCost(uint256 billId) external view returns (euint32) {
        return bills[billId].encryptedCost;
    }

    // -----------------------
    // Safety / Governance notes
    // -----------------------
    // - This contract avoids storing plaintext sensitive data permanently. When decryption callbacks
    //   happen, integrators should be careful about where plaintexts are emitted / stored (events are
    //   immutable and public).
    // - Access control for who may initiate decryption / training / billing must be adapted to the
    //   governance model: e.g., multisig for insurer payouts, auditor multisig for certain reveals.
    // - Differential privacy integration: here we enforce a simple on-chain epsilon upper-bound, but
    //   real DP mechanisms (noise injection) should be applied by the FHE runtime during computation,
    //   or by combining FHE computations with DP noise ciphertexts.
    // - This contract treats euint32 as a placeholder encrypted numeric type from the FHE library.
    //   In practice, encrypted vectors/tensors are often represented as arrays of ciphertexts (bytes32[]).
}
