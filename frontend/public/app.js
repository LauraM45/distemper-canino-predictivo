/**
 * Lógica de interacción para el Sistema Predictivo de Distemper Canino (CDV).
 * Consume los endpoints de FastAPI y gestiona la reactividad de la interfaz.
 */

const API_BASE = ''; // Relativo al mismo host (http://localhost:8000)

let chartRiesgoInstance = null;
let chartSignosInstance = null;

// ================= GESTIÓN DE PESTAÑAS =================
function cambiarPestana(nombre) {
  document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

  const targetSec = document.getElementById(`section-${nombre}`);
  const targetBtn = document.getElementById(`tab-${nombre.substring(0, 4)}-btn`);

  if (targetSec) targetSec.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');

  if (nombre === 'dashboard') {
    cargarMetricasDashboard();
  } else if (nombre === 'historial') {
    cargarHistorialPacientes();
  }
}

// ================= ENVÍO DEL FORMULARIO Y PREDICCIÓN =================
async function enviarDiagnostico(event) {
  event.preventDefault();

  const btnSubmit = document.getElementById('btn-submit');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');
  const resultCard = document.getElementById('result-container');

  // Activar estado de carga
  btnSubmit.disabled = true;
  btnText.textContent = 'Analizando paciente con ML...';
  btnSpinner.style.display = 'inline-block';

  // Extraer datos del formulario
  const form = document.getElementById('cdv-form');
  const formData = new FormData(form);

  const payload = {
    edad_meses: parseFloat(formData.get('edad_meses')),
    sexo: formData.get('sexo'),
    raza: formData.get('raza'),
    talla: formData.get('talla'),
    ubicacion_procedencia: formData.get('ubicacion_procedencia'),
    estado_vacunal: formData.get('estado_vacunal'),
    fiebre_hipertermia: formData.get('fiebre_hipertermia') === '1' ? 1 : 0,
    signos_respiratorios_oculonasales: formData.get('signos_respiratorios_oculonasales') === '1' ? 1 : 0,
    signos_digestivos: formData.get('signos_digestivos') === '1' ? 1 : 0,
    signos_neurologicos: formData.get('signos_neurologicos') === '1' ? 1 : 0,
    signos_dermatologicos: formData.get('signos_dermatologicos') === '1' ? 1 : 0,
    notas_veterinarias: formData.get('notas_veterinarias') || null,
  };

  try {
    const response = await fetch(`${API_BASE}/api/evaluar/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error en la inferencia del modelo.');
    }

    const data = await response.json();
    mostrarResultado(data);
  } catch (error) {
    alert(`Error al generar diagnóstico: ${error.message}`);
  } finally {
    btnSubmit.disabled = false;
    btnText.textContent = '🔬 Generar Diagnóstico Predictivo';
    btnSpinner.style.display = 'none';
  }
}

// ================= MOSTRAR RESULTADO INMEDIATO =================
function mostrarResultado(data) {
  const resultCard = document.getElementById('result-container');
  const riskBadge = document.getElementById('risk-badge');
  const probPct = document.getElementById('prob-percentage');
  const progressFill = document.getElementById('progress-bar-fill');
  const diagText = document.getElementById('diagnosis-text');
  const actionText = document.getElementById('clinical-action-text');

  const pct = Math.round(data.probability_cdv * 100);
  probPct.textContent = `${pct}%`;
  progressFill.style.width = `${pct}%`;

  // Colores por nivel de riesgo
  riskBadge.className = 'risk-badge';
  if (data.risk_level === 'Alto Riesgo') {
    riskBadge.classList.add('high');
    progressFill.style.backgroundColor = '#ef4444';
  } else if (data.risk_level === 'Riesgo Moderado') {
    riskBadge.classList.add('moderate');
    progressFill.style.backgroundColor = '#f59e0b';
  } else {
    riskBadge.classList.add('low');
    progressFill.style.backgroundColor = '#10b981';
  }

  riskBadge.textContent = data.risk_level.toUpperCase();
  diagText.textContent = data.diagnosis;
  actionText.textContent = data.clinical_action;

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function cerrarResultado() {
  document.getElementById('result-container').style.display = 'none';
  document.getElementById('cdv-form').reset();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ================= CARGA DE MÉTRICAS DEL DASHBOARD =================
async function cargarMetricasDashboard() {
  try {
    const response = await fetch(`${API_BASE}/api/dashboard/metricas`);
    if (!response.ok) throw new Error('No se pudieron obtener métricas');

    const data = await response.json();
    const resumen = data.resumen;

    // Actualizar KPIs
    document.getElementById('kpi-total').textContent = resumen.total_evaluados;
    document.getElementById('kpi-positivos').textContent = resumen.total_positivos;
    document.getElementById('kpi-tasa').textContent = `${resumen.tasa_positividad}% tasa de positividad`;
    document.getElementById('kpi-alto-riesgo').textContent = resumen.alto_riesgo_count;
    document.getElementById('kpi-sanos').textContent = resumen.bajo_riesgo_count;

    // Renderizar Gráfico 1: Distribución de Riesgo (Dona)
    renderGraficoRiesgo(data.distribucion_riesgo);

    // Renderizar Gráfico 2: Prevalencia de Signos (Barras)
    renderGraficoSignos(data.prevalencia_signos);
  } catch (error) {
    console.error('Error al cargar dashboard:', error);
  }
}

function renderGraficoRiesgo(distribucion) {
  const ctx = document.getElementById('chart-riesgo').getContext('2d');
  if (chartRiesgoInstance) chartRiesgoInstance.destroy();

  chartRiesgoInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Alto Riesgo', 'Riesgo Moderado', 'Bajo Riesgo'],
      datasets: [
        {
          data: [
            distribucion['Alto Riesgo'] || 0,
            distribucion['Riesgo Moderado'] || 0,
            distribucion['Bajo Riesgo'] || 0,
          ],
          backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
          borderWidth: 2,
          borderColor: '#1e293b',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#e2e8f0', font: { family: 'Outfit', size: 12 } },
        },
      },
    },
  });
}

function renderGraficoSignos(signos) {
  const ctx = document.getElementById('chart-signos').getContext('2d');
  if (chartSignosInstance) chartSignosInstance.destroy();

  const labels = signos.map(s => s.etiqueta.split('(')[0].trim());
  const dataPositivos = signos.map(s => s.porcentaje_en_positivos);

  chartSignosInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: '% en Casos Positivos CDV',
          data: dataPositivos,
          backgroundColor: 'rgba(239, 68, 68, 0.75)',
          borderColor: '#ef4444',
          borderWidth: 1,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: '#94a3b8', callback: val => `${val}%` },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
        },
        x: {
          ticks: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } },
          grid: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

// ================= HISTORIAL DE PACIENTES =================
async function cargarHistorialPacientes() {
  const tbody = document.getElementById('pacientes-table-body');
  try {
    const res = await fetch(`${API_BASE}/api/pacientes/?limit=50`);
    if (!res.ok) throw new Error('Error al cargar historial');

    const pacientes = await res.json();
    if (pacientes.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--text-muted); padding: 2rem;">No hay pacientes registrados aún en la base de datos.</td></tr>`;
      return;
    }

    tbody.innerHTML = pacientes
      .map(p => {
        const fecha = new Date(p.fecha_registro).toLocaleDateString('es-CO', {
          day: '2-digit',
          month: 'short',
          hour: '2-digit',
          minute: '2-digit',
        });
        const prob = Math.round(p.probabilidad_cdv * 100);
        const badgeClass =
          p.nivel_riesgo === 'Alto Riesgo' ? 'alto' : p.nivel_riesgo === 'Riesgo Moderado' ? 'moderado' : 'bajo';

        return `
        <tr>
          <td>${fecha}</td>
          <td><strong>${p.raza}</strong></td>
          <td>${p.edad_meses} m</td>
          <td>${p.sexo}</td>
          <td>${p.talla}</td>
          <td>${p.prediccion_cdv === 1 ? '🚨 Positivo' : '✅ Sano'}</td>
          <td>${prob}%</td>
          <td><span class="table-badge ${badgeClass}">${p.nivel_riesgo}</span></td>
        </tr>
      `;
      })
      .join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: #f87171;">Error al consultar la base de datos: ${err.message}</td></tr>`;
  }
}

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
  // Comprobar estado de la API
  fetch(`${API_BASE}/api/health`)
    .then(r => r.json())
    .then(data => {
      document.getElementById('db-status').textContent = '● Neon DB Conectado';
    })
    .catch(() => {
      document.getElementById('db-status').textContent = '○ Conectando al Servidor...';
      document.getElementById('db-status').style.color = '#fbbf24';
    });
});
