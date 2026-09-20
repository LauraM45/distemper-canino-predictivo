import React from 'react';

export default function RiskGauge({ probability, riskLevel }) {
  const percentage = Math.round((probability || 0) * 100);

  const getBadgeClass = () => {
    if (riskLevel === 'Alto Riesgo') return 'risk-badge high';
    if (riskLevel === 'Riesgo Moderado') return 'risk-badge moderate';
    return 'risk-badge low';
  };

  const getBarColor = () => {
    if (riskLevel === 'Alto Riesgo') return '#ef4444';
    if (riskLevel === 'Riesgo Moderado') return '#f59e0b';
    return '#10b981';
  };

  return (
    <div className="gauge-wrapper">
      <div className="gauge-metric">
        <span className="metric-number">{percentage}%</span>
        <span className={getBadgeClass()}>{riskLevel?.toUpperCase()}</span>
      </div>
      <div className="progress-bar-bg">
        <div
          className="progress-bar-fill"
          style={{ width: `${percentage}%`, backgroundColor: getBarColor() }}
        />
      </div>
      <small className="hint" style={{ marginTop: '0.5rem', display: 'block' }}>
        Probabilidad estimada por el modelo de Scikit-Learn (Recall clínico garantizado).
      </small>
    </div>
  );
}
