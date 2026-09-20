/**
 * Cliente API para la comunicación entre React y FastAPI.
 */

const API_BASE_URL = import.meta.env?.VITE_API_URL || 'http://localhost:8000';

export async function evaluarPaciente(datosPaciente) {
  const res = await fetch(`${API_BASE_URL}/api/evaluar/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(datosPaciente),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Error al evaluar el paciente');
  }
  return await res.json();
}

export async function obtenerMetricasDashboard() {
  const res = await fetch(`${API_BASE_URL}/api/dashboard/metricas`);
  if (!res.ok) throw new Error('Error al obtener métricas del dashboard');
  return await res.json();
}

export async function listarPacientes(limit = 50) {
  const res = await fetch(`${API_BASE_URL}/api/pacientes/?limit=${limit}`);
  if (!res.ok) throw new Error('Error al listar pacientes');
  return await res.json();
}
