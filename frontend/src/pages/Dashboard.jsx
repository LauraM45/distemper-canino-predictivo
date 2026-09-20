import React, { useEffect, useState } from 'react';
import { obtenerMetricasDashboard } from '../api/client';

export default function Dashboard() {
  const [metricas, setMetricas] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchMetricas = async () => {
    setLoading(true);
    try {
      const data = await obtenerMetricasDashboard();
      setMetricas(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetricas();
  }, []);

  if (loading) return <div className="card"><p>Cargando métricas del Dashboard...</p></div>;
  if (!metricas) return <div className="card"><p>No hay datos disponibles en la base de datos.</p></div>;

  const { resumen, distribucion_riesgo, prevalencia_signos } = metricas;

  return (
    <div>
      <div className="dash-header">
        <div>
          <h2>Dashboard Epidemiológico</h2>
          <p>Métricas consolidadas en tiempo real desde Neon DB.</p>
        </div>
        <button className="btn-secondary" onClick={fetchMetricas}>🔄 Refrescar</button>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <span class="kpi-label">Total Evaluados</span>
          <span class="kpi-value">{resumen.total_evaluados}</span>
          <span class="kpi-sub">Caninos analizados</span>
        </div>
        <div className="kpi-card kpi-positive">
          <span class="kpi-label">Casos Positivos CDV</span>
          <span class="kpi-value">{resumen.total_positivos}</span>
          <span class="kpi-sub">{resumen.tasa_positividad}% positividad</span>
        </div>
        <div className="kpi-card kpi-warning">
          <span class="kpi-label">Alto Riesgo</span>
          <span class="kpi-value">{resumen.alto_riesgo_count}</span>
          <span class="kpi-sub">Requieren aislamiento</span>
        </div>
        <div className="kpi-card kpi-healthy">
          <span class="kpi-label">Bajo Riesgo / Sanos</span>
          <span class="kpi-value">{resumen.bajo_riesgo_count}</span>
          <span class="kpi-sub">Manejo preventivo</span>
        </div>
      </div>

      <div className="charts-grid">
        <div className="card chart-card">
          <h3>Estratificación de Riesgo</h3>
          <p className="chart-desc">Distribución según probabilidad de infección.</p>
          <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>
            <li style={{ padding: '0.5rem 0', color: '#ef4444' }}>🚨 Alto Riesgo: {distribucion_riesgo['Alto Riesgo']}</li>
            <li style={{ padding: '0.5rem 0', color: '#f59e0b' }}>⚠️ Riesgo Moderado: {distribucion_riesgo['Riesgo Moderado']}</li>
            <li style={{ padding: '0.5rem 0', color: '#10b981' }}>✅ Bajo Riesgo: {distribucion_riesgo['Bajo Riesgo']}</li>
          </ul>
        </div>

        <div className="card chart-card">
          <h3>Prevalencia de Signos en Positivos</h3>
          <p className="chart-desc">Porcentaje de presencia en casos CDV confirmados.</p>
          <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>
            {prevalencia_signos.map((s) => (
              <li key={s.signo_clave} style={{ padding: '0.4rem 0', display: 'flex', justifyContent: 'space-between' }}>
                <span>{s.etiqueta}</span>
                <strong>{s.porcentaje_en_positivos}%</strong>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
