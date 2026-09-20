import React, { useState } from 'react';
import FormularioDiagnostico from './pages/FormularioDiagnostico';
import Dashboard from './pages/Dashboard';

export default function App() {
  const [activeTab, setActiveTab] = useState('formulario');

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="nav-container">
          <div className="brand">
            <div className="brand-icon">🐕‍🦺</div>
            <div className="brand-info">
              <h1>CDV Predict</h1>
              <p>Sistema Clínico de Soporte al Diagnóstico de Distemper Canino</p>
            </div>
          </div>
          <div className="nav-badges">
            <span className="badge badge-ml">ML Pipeline Scikit-Learn</span>
            <span className="badge badge-db">● Neon DB Conectado</span>
          </div>
        </div>

        <div className="tabs-container">
          <button
            className={`tab-btn ${activeTab === 'formulario' ? 'active' : ''}`}
            onClick={() => setActiveTab('formulario')}
          >
            🩺 Nuevo Diagnóstico
          </button>
          <button
            className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard Estadístico
          </button>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'formulario' && <FormularioDiagnostico onVerDashboard={() => setActiveTab('dashboard')} />}
        {activeTab === 'dashboard' && <Dashboard />}
      </main>
    </div>
  );
}
