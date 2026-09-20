import React, { useState } from 'react';
import { evaluarPaciente } from '../api/client';
import RiskGauge from '../components/RiskGauge';

export default function FormularioDiagnostico({ onVerDashboard }) {
  const [formData, setFormData] = useState({
    edad_meses: '',
    sexo: 'Macho',
    raza: '',
    talla: 'Mediano',
    ubicacion_procedencia: '',
    estado_vacunal: 'Incompleto',
    fiebre_hipertermia: 0,
    signos_respiratorios_oculonasales: 0,
    signos_digestivos: 0,
    signos_neurologicos: 0,
    signos_dermatologicos: 0,
    notas_veterinarias: '',
  });

  const [loading, setLoading] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = {
        ...formData,
        edad_meses: parseFloat(formData.edad_meses),
      };
      const res = await evaluarPaciente(payload);
      setResultado(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card form-card">
      <div className="card-header">
        <h2>Evaluación Clínica del Paciente Canino</h2>
        <p>Ingrese las 11 variables para generar la predicción de riesgo de Moquillo Canino (CDV).</p>
      </div>

      {error && <div className="detail-box" style={{ borderLeftColor: '#ef4444', marginBottom: '1.5rem', color: '#f87171' }}>{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label>Edad (en meses) *</label>
            <input
              type="number"
              name="edad_meses"
              step="0.1"
              min="0.5"
              max="240"
              value={formData.edad_meses}
              onChange={handleChange}
              placeholder="Ej: 4.5"
              required
            />
          </div>

          <div className="form-group">
            <label>Sexo *</label>
            <select name="sexo" value={formData.sexo} onChange={handleChange}>
              <option value="Macho">Macho</option>
              <option value="Hembra">Hembra</option>
            </select>
          </div>

          <div className="form-group">
            <label>Raza *</label>
            <input
              type="text"
              name="raza"
              value={formData.raza}
              onChange={handleChange}
              placeholder="Ej: Criollo, Labrador, Poodle..."
              required
            />
          </div>

          <div className="form-group">
            <label>Talla *</label>
            <select name="talla" value={formData.talla} onChange={handleChange}>
              <option value="Pequeño">Pequeño</option>
              <option value="Mediano">Mediano</option>
              <option value="Grande">Grande</option>
              <option value="Gigante">Gigante</option>
            </select>
          </div>

          <div className="form-group">
            <label>Centro / Procedencia *</label>
            <input
              type="text"
              name="ubicacion_procedencia"
              value={formData.ubicacion_procedencia}
              onChange={handleChange}
              placeholder="Ej: Clínica Norte, Albergue..."
              required
            />
          </div>

          <div className="form-group">
            <label>Estado Vacunal *</label>
            <select name="estado_vacunal" value={formData.estado_vacunal} onChange={handleChange}>
              <option value="Completo">Completo</option>
              <option value="Incompleto">Incompleto</option>
              <option value="No vacunado">No vacunado</option>
              <option value="Desconocido">Desconocido</option>
            </select>
          </div>
        </div>

        {/* Interruptores de Signos Clínicos */}
        <div className="signs-section">
          <h3>Signos Clínicos Observados</h3>
          <p className="signs-desc">Active los interruptores correspondientes:</p>

          <div className="signs-grid">
            <label className="sign-item">
              <input
                type="checkbox"
                name="fiebre_hipertermia"
                checked={formData.fiebre_hipertermia === 1}
                onChange={handleChange}
              />
              <span className="sign-slider"></span>
              <span className="sign-label">
                <strong>🌡️ Fiebre / Hipertermia</strong>
                <small>&gt; 39.2 °C</small>
              </span>
            </label>

            <label className="sign-item">
              <input
                type="checkbox"
                name="signos_respiratorios_oculonasales"
                checked={formData.signos_respiratorios_oculonasales === 1}
                onChange={handleChange}
              />
              <span className="sign-slider"></span>
              <span className="sign-label">
                <strong>👃 Signos Oculonasales</strong>
                <small>Descarga o tos</small>
              </span>
            </label>

            <label className="sign-item">
              <input
                type="checkbox"
                name="signos_digestivos"
                checked={formData.signos_digestivos === 1}
                onChange={handleChange}
              />
              <span className="sign-slider"></span>
              <span className="sign-label">
                <strong>🤢 Signos Digestivos</strong>
                <small>Vómito o diarrea</small>
              </span>
            </label>

            <label className="sign-item">
              <input
                type="checkbox"
                name="signos_neurologicos"
                checked={formData.signos_neurologicos === 1}
                onChange={handleChange}
              />
              <span className="sign-slider"></span>
              <span className="sign-label">
                <strong>⚡ Signos Neurológicos</strong>
                <small>Mioclonías o ataxia</small>
              </span>
            </label>

            <label className="sign-item">
              <input
                type="checkbox"
                name="signos_dermatologicos"
                checked={formData.signos_dermatologicos === 1}
                onChange={handleChange}
              />
              <span className="sign-slider"></span>
              <span className="sign-label">
                <strong>🐾 Signos Dermatológicos</strong>
                <small>Hiperqueratosis</small>
              </span>
            </label>
          </div>
        </div>

        <div className="form-actions">
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Analizando con ML...' : '🔬 Generar Diagnóstico Predictivo'}
          </button>
        </div>
      </form>

      {/* Resultado */}
      {resultado && (
        <div className="card result-card" style={{ marginTop: '2rem' }}>
          <h3>Diagnóstico Emitido</h3>
          <RiskGauge probability={resultado.probability_cdv} riskLevel={resultado.risk_level} />
          <div className="result-details" style={{ marginTop: '1.5rem' }}>
            <div className="detail-box">
              <strong>Diagnóstico:</strong>
              <p>{resultado.diagnosis}</p>
            </div>
            <div className="detail-box">
              <strong>Acción Clínica Sugerida:</strong>
              <p>{resultado.clinical_action}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
