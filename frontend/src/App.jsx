import { useState } from "react";
import "./App.css";

// Configurable API base URL: falls back to localhost:8000 for local dev.
// Set VITE_API_URL in a .env file (or the environment) to point the
// frontend at a different backend host/port.
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const ANALYZE_URL = `${API_BASE_URL}/api/v1/analyze`;

// How long to wait for /analyze before giving up and telling the user,
// instead of leaving the button stuck on "Analyzing..." forever.
const ANALYZE_TIMEOUT_MS = 30_000;

function App() {
  const [responseText, setResponseText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeResponse = async () => {
    if (!responseText.trim()) {
      setError("Please enter an LLM response.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      ANALYZE_TIMEOUT_MS
    );

    try {
      const response = await fetch(ANALYZE_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          response_text: responseText,
        }),
        signal: controller.signal,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed");
      }

      setResult(data);
    } catch (err) {
      if (err.name === "AbortError") {
        setError(
          "The analysis is taking longer than expected " +
            `(over ${ANALYZE_TIMEOUT_MS / 1000}s) and was cancelled. ` +
            "The backend may be slow to start up or unreachable at " +
            `${ANALYZE_URL}. Please try again.`
        );
      } else {
        setError(err.message || "Failed to connect to backend.");
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  // Get the evidence array for a claim. Evidence items are objects of the
  // shape { text, source, similarity_score }, matching the backend's
  // EvidenceItem contract (see backend/app/api/analysis.py).
  const getClaimEvidence = (claim) => {
    if (Array.isArray(claim.evidence)) {
      return claim.evidence;
    }

    return [];
  };

  // Human-readable label for a claim's verification_status.
  const formatVerificationStatus = (status) => {
    switch (status) {
      case "supported":
        return "✅ Supported";
      case "contradicted":
        return "❌ Contradicted";
      case "insufficient_evidence":
        return "⚠️ Insufficient Evidence";
      case "unverified":
        return "❔ Unverified";
      default:
        return status || "N/A";
    }
  };

  // Similarity scores come from the backend as a 0-1 float; render them
  // consistently rounded to 2 decimal places.
  const formatSimilarityScore = (score) => {
    if (typeof score !== "number" || Number.isNaN(score)) {
      return null;
    }
    return score.toFixed(2);
  };

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <h1>Hallucination Detection System</h1>
        <p>AI-powered LLM Response Analysis</p>
      </header>

      <main className="container">

        {/* ANALYZE SECTION */}
        <section className="card">
          <h2>Analyze LLM Response</h2>

          <textarea
            value={responseText}
            onChange={(e) => setResponseText(e.target.value)}
            placeholder="Enter an LLM response here..."
            rows="8"
          />

          <button
            onClick={analyzeResponse}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze Response"}
          </button>

          {error && (
            <div className="error">
              ❌ {error}
            </div>
          )}
        </section>

        {/* RESULT SECTION */}
        {result && (
          <section className="card result">

            <h2>Analysis Result</h2>

            {/* VERDICT */}
            <div className="verdict">
              <h3>Verdict</h3>
              <p>{result.verdict || "Not available"}</p>
            </div>

            {/* SCORES */}
            {result.scores && (
              <div className="scores">

                <div className="score-box">
                  <span>Trust Score</span>
                  <strong>
                    {result.scores.trust_score ?? "N/A"}
                  </strong>
                </div>

                <div className="score-box">
                  <span>Reliability Score</span>
                  <strong>
                    {result.scores.reliability_score ?? "N/A"}
                  </strong>
                </div>

                <div className="score-box">
                  <span>Hallucination Probability</span>
                  <strong>
                    {result.scores.hallucination_probability ?? "N/A"}
                  </strong>
                </div>

                <div className="score-box">
                  <span>Confidence Score</span>
                  <strong>
                    {result.scores.confidence_score ?? "N/A"}
                  </strong>
                </div>

                <div className="score-box">
                  <span>Risk Level</span>
                  <strong>
                    {result.scores.risk_level ?? "N/A"}
                  </strong>
                </div>

              </div>
            )}

            {/* CLAIM SUMMARY */}
            {result.claim_summary && (
              <div className="claim-summary">

                <h3>Claim Summary</h3>

                <p>
                  <strong>Total Claims:</strong>{" "}
                  {result.claim_summary.total_claims ?? 0}
                </p>

                <p>
                  <strong>Supported:</strong>{" "}
                  {result.claim_summary.supported ?? 0}
                </p>

                <p>
                  <strong>Contradicted:</strong>{" "}
                  {result.claim_summary.contradicted ?? 0}
                </p>

                <p>
                  <strong>Insufficient Evidence:</strong>{" "}
                  {result.claim_summary.insufficient_evidence ?? 0}
                </p>

              </div>
            )}

            {/* CLAIMS */}
            {Array.isArray(result.claims) && result.claims.length > 0 && (
              <div className="claims">

                <h3>Claims</h3>

                {result.claims.map((claim, index) => {

                  const evidence = getClaimEvidence(claim);

                  return (
                    <div
                      className="claim"
                      key={claim.id || index}
                    >

                      {/* CLAIM TEXT */}
                      <p>
                        <strong>Claim:</strong>{" "}
                        {claim.text || "No claim text available"}
                      </p>

                      {/* CLAIM TYPE */}
                      <p>
                        <strong>Type:</strong>{" "}
                        {claim.claim_type || "N/A"}
                      </p>

                      {/* VERIFICATION STATUS */}
                      <p>
                        <strong>Verification Status:</strong>{" "}
                        <span className="claim-status">
                          {formatVerificationStatus(
                            claim.verification_status
                          )}
                        </span>
                      </p>

                      {/* EXPLANATION */}
                      {claim.explanation && (
                        <p>
                          <strong>Explanation:</strong>{" "}
                          {claim.explanation}
                        </p>
                      )}

                      {/* EVIDENCE */}
                      <div className="evidence">

                        <h4>🔍 Evidence</h4>

                        {evidence.length > 0 ? (

                          evidence.map((item, evidenceIndex) => (
                            <div
                              className="evidence-item"
                              key={evidenceIndex}
                            >

                              <p>
                                <strong>Source:</strong>{" "}
                                {item.source || "Unknown source"}
                              </p>

                              {item.text && (
                                <p className="evidence-text">
                                  {item.text}
                                </p>
                              )}

                              {formatSimilarityScore(
                                item.similarity_score
                              ) !== null && (
                                <p>
                                  <strong>Similarity Score:</strong>{" "}
                                  {formatSimilarityScore(
                                    item.similarity_score
                                  )}
                                </p>
                              )}

                            </div>
                          ))

                        ) : (

                          <p className="no-evidence">
                            No evidence available for this claim.
                          </p>

                        )}

                      </div>

                    </div>
                  );
                })}

              </div>
            )}

            {/* RAW JSON */}
            <details>
              <summary>View Raw JSON Response</summary>

              <pre>
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>

          </section>
        )}

      </main>
    </div>
  );
}

export default App;