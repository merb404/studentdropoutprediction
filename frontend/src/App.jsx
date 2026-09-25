import React, { useState } from "react";

const API_BASE = "http://localhost:5000";

const initialForm = {
  admission_grade: 120,
  sem1_grade: 12,
  sem2_grade: 12,
  sem2_approved: 5,
  age: 19,
  scholarship: 0,
  tuition_up_to_date: 1,
  debtor: 0,
};

function RiskStamp({ result }) {
  if (!result) {
    return (
      <div className="stamp stamp--empty">
        <span>No assessment yet</span>
      </div>
    );
  }

  const level = result.risk_level.split(" ")[0].toLowerCase(); // low / medium / high

  return (
    <div className={`stamp stamp--${level} stamp--enter`} key={result.probability_pct}>
      <span className="stamp__level">{result.risk_level}</span>
      <span className="stamp__prediction">{result.prediction}</span>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function update(key, value) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function handleAssess(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      setResult(data);
      setHistory((h) => [
        {
          id: h.length + 1,
          admission_grade: form.admission_grade,
          sem1_grade: form.sem1_grade,
          sem2_grade: form.sem2_grade,
          sem2_approved: form.sem2_approved,
          age: form.age,
          probability_pct: data.probability_pct,
          risk_level: data.risk_level,
        },
        ...h,
      ]);
    } catch (err) {
      setError(
        "Couldn't reach the prediction server. Make sure the Flask backend is running on http://localhost:5000 (python app.py)."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="desk">
      <header className="masthead">
        <div className="masthead__mark">§</div>
        <div>
          <h1>Registrar&rsquo;s Risk Desk</h1>
          <p className="masthead__sub">Dropout-risk assessment, on file per student</p>
        </div>
      </header>

      <main className="deskbody">
        <form className="intake" onSubmit={handleAssess}>
          <h2>Intake form</h2>

          <div className="field-row">
            <label htmlFor="admission_grade">Admission grade</label>
            <input
              id="admission_grade"
              type="number" min="0" max="200"
              value={form.admission_grade}
              onChange={(e) => update("admission_grade", Number(e.target.value))}
            />
          </div>

          <div className="field-row">
            <label htmlFor="sem1_grade">1st semester grade</label>
            <input
              id="sem1_grade"
              type="number" min="0" max="20" step="0.1"
              value={form.sem1_grade}
              onChange={(e) => update("sem1_grade", Number(e.target.value))}
            />
          </div>

          <div className="field-row">
            <label htmlFor="sem2_grade">2nd semester grade</label>
            <input
              id="sem2_grade"
              type="number" min="0" max="20" step="0.1"
              value={form.sem2_grade}
              onChange={(e) => update("sem2_grade", Number(e.target.value))}
            />
          </div>

          <div className="field-row">
            <label htmlFor="sem2_approved">2nd sem. units approved</label>
            <input
              id="sem2_approved"
              type="number" min="0" max="20"
              value={form.sem2_approved}
              onChange={(e) => update("sem2_approved", Number(e.target.value))}
            />
          </div>

          <div className="field-row">
            <label htmlFor="age">Age at enrollment</label>
            <input
              id="age"
              type="number" min="16" max="70"
              value={form.age}
              onChange={(e) => update("age", Number(e.target.value))}
            />
          </div>

          <div className="toggles">
            <label className="toggle">
              <input
                type="checkbox"
                checked={form.scholarship === 1}
                onChange={(e) => update("scholarship", e.target.checked ? 1 : 0)}
              />
              Scholarship holder
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={form.tuition_up_to_date === 1}
                onChange={(e) => update("tuition_up_to_date", e.target.checked ? 1 : 0)}
              />
              Tuition fees up to date
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={form.debtor === 1}
                onChange={(e) => update("debtor", e.target.checked ? 1 : 0)}
              />
              Flagged as debtor
            </label>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Assessing…" : "Assess risk"}
          </button>

          {error && <p className="error-note">{error}</p>}
        </form>

        <section className="assessment">
          <h2>Assessment</h2>

          <div className="probability-block">
            <span className="probability-block__label">Dropout probability</span>
            <span className="probability-block__value">
              {result ? `${result.probability_pct}%` : "—"}
            </span>
          </div>

          <RiskStamp result={result} />

          {result && (
            <p className="model-note">
              Model: Logistic Regression &middot; trained on{" "}
              {result.input_echo ? "your live dataset pipeline" : "—"} at server startup.
            </p>
          )}
        </section>
      </main>

      <section className="ledger">
        <h2>Case log</h2>
        {history.length === 0 ? (
          <p className="ledger__empty">Assessed cases will be logged here, most recent first.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>No.</th>
                <th>Admission</th>
                <th>Sem 1</th>
                <th>Sem 2</th>
                <th>Approved</th>
                <th>Age</th>
                <th>P(Dropout)</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {history.map((row) => (
                <tr key={row.id}>
                  <td>{row.id}</td>
                  <td>{row.admission_grade}</td>
                  <td>{row.sem1_grade}</td>
                  <td>{row.sem2_grade}</td>
                  <td>{row.sem2_approved}</td>
                  <td>{row.age}</td>
                  <td>{row.probability_pct}%</td>
                  <td>{row.risk_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
