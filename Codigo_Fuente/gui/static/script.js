document.addEventListener('DOMContentLoaded', async () => {
    const dem_path = document.getElementById('dem_path');
    const mode = document.getElementById('mode');
    const output_dir = document.getElementById('output_dir');
    const pca_component = document.getElementById('pca_component');
    const z_factor = document.getElementById('z_factor');
    const include_slope = document.getElementById('include_slope');
    const include_curvature = document.getElementById('include_curvature');
    const detector = document.getElementById('detector');
    const lthr = document.getElementById('lthr');
    const fthr = document.getElementById('fthr');
    const athr = document.getElementById('athr');
    const dthr = document.getElementById('dthr');
    const gthr = document.getElementById('gthr');
    const gthr_value = document.getElementById('gthr-value');
    const min_length = document.getElementById('min_length');
    const dilation_radius = document.getElementById('dilation_radius');
    const btn_run = document.getElementById('btn-run');
    const terminal = document.getElementById('terminal');

    gthr.addEventListener('input', (e) => { gthr_value.textContent = e.target.value; });

    // Cargar config inicial
    try {
        const response = await fetch('/api/config');
        const c = await response.json();
        
        dem_path.value = c.DEM_PATH;
        mode.value = c.MODE;
        output_dir.value = c.OUTPUT_DIR;
        pca_component.value = c.PCA_COMPONENT;
        z_factor.value = c.Z_FACTOR;
        include_slope.checked = c.INCLUDE_SLOPE;
        include_curvature.checked = c.INCLUDE_CURVATURE;
        detector.value = c.DETECTOR;
        lthr.value = c.LTHR;
        fthr.value = c.FTHR;
        athr.value = c.ATHR;
        dthr.value = c.DTHR;
        gthr.value = c.GTHR_PERCENTILE;
        gthr_value.textContent = c.GTHR_PERCENTILE;
        min_length.value = c.MIN_LENGTH_M;
        dilation_radius.value = c.DILATION_RADIUS;

        addLog(`Configuración cargada correctamente.`, 'success');
    } catch (err) {
        addLog(`Error cargando config: ${err.message}`, 'warn');
    }

    btn_run.addEventListener('click', async () => {
        btn_run.disabled = true;
        btn_run.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Procesando...';
        lucide.createIcons();

        const payload = {
            dem_path: dem_path.value,
            mode: mode.value,
            output_dir: output_dir.value,
            pca_component: pca_component.value,
            z_factor: z_factor.value,
            include_slope: include_slope.checked,
            include_curvature: include_curvature.checked,
            detector: detector.value,
            lthr: parseInt(lthr.value),
            fthr: parseInt(fthr.value),
            athr: parseInt(athr.value),
            dthr: parseInt(dthr.value),
            gthr_percentile: parseInt(gthr.value),
            min_length_m: parseInt(min_length.value),
            dilation_radius: parseInt(dilation_radius.value)
        };

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                addLog(`Proceso iniciado...`, 'success');
                const eventSource = new EventSource(`/api/logs`);
                eventSource.onmessage = (event) => {
                    if (event.data === '[EOF]') {
                        eventSource.close();
                        addLog('Procesamiento completado con éxito.', 'success');
                        btn_run.disabled = false;
                        btn_run.innerHTML = '<i data-lucide="play"></i> Iniciar Proceso';
                        lucide.createIcons();

                        // Actualizar Vista Previa
                        const previewContainer = document.getElementById('preview-container');
                        previewContainer.innerHTML = `<img src="/api/preview?t=${new Date().getTime()}" style="max-width:100%; max-height:100%; border-radius:1rem;">`;
                    } else {
                        addLog(event.data, 'info');
                    }
                };
            }
        } catch (err) {
            addLog(`Error: ${err.message}`, 'warn');
            btn_run.disabled = false;
        }
    });

    function addLog(msg, type = 'info') {
        const span = document.createElement('span');
        span.className = type;
        const time = new Date().toLocaleTimeString();
        span.innerHTML = `[${time}] ${msg}<br>`;
        terminal.appendChild(span);
        terminal.scrollTop = terminal.scrollHeight;
    }
});
