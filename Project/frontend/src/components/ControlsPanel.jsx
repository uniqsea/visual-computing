import { useCallback, useEffect, useState } from 'react';
import styled from 'styled-components';
import { uploadSketch, vectorizeRaster } from '../services/api';
import SketchCanvas from './SketchCanvas';
import { Upload } from 'lucide-react';

// New modular components
import Navbar from './controls/Navbar';
import SegmentedControl from './controls/SegmentedControl';
import ImageUpload from './controls/ImageUpload';
import ExtrusionControls from './controls/ExtrusionControls';
import RevolutionControls from './controls/RevolutionControls';
import HeightMapControls from './controls/HeightMapControls';

const RUN_MODES = ['heightmap', 'extrusion', 'revolution'];

const createEmptyResults = () => ({
  heightmap: { status: 'pending' },
  extrusion: { status: 'pending' },
  revolution: { status: 'pending' }
});

const cloneSettings = (settings = {}) => ({
  extrusion: { ...(settings.extrusion || {}) },
  revolution: { ...(settings.revolution || {}) },
  heightmap: { ...(settings.heightmap || {}) }
});

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
`;

const Panel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  overflow-y: auto;
  flex: 1;
`;

const HeaderRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
`;

const Title = styled.h2`
  font-size: 1rem;
  font-weight: 600;
  color: ${(props) => props.theme.colors.accent};
`;

const Button = styled.button`
  padding: 16px;
  border: none;
  background: ${(props) => props.theme.colors.accent};
  color: #000000;
  font-weight: 700;
  font-size: 1rem;
  border-radius: ${(props) => props.theme.radii.lg};
  cursor: pointer;
  transition: all 0.2s ease;
  margin-top: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;

  &:hover {
    opacity: 0.9;
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }
`;

const StatusMessage = styled.p`
  color: ${(props) => (props.$isError ? '#f87171' : props.theme.colors.text)};
  font-size: 0.875rem;
  margin-top: 8px;
`;

const BatchCard = styled.div`
  padding: 16px;
  border-radius: 12px;
  border: 1px solid ${(props) => props.theme.colors.border};
  background: #1f1f23;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const InlineActions = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const SecondaryButton = styled.button`
  padding: 10px 12px;
  border-radius: ${(props) => props.theme.radii.sm};
  border: 1px solid ${(props) => props.theme.colors.border};
  background: transparent;
  color: ${(props) => props.theme.colors.text};
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ProgressText = styled.div`
  font-size: 0.85rem;
  color: #a1a1aa;
`;

function ControlsPanel({
  mode, setMode,
  evaluationMode, setEvaluationMode,
  evaluationFiles, setEvaluationFiles,
  selectedEvalIndex, setSelectedEvalIndex,
  onSettingsChange
}) {
  // const [mode, setMode] = useState('heightmap'); // Lifted to App.jsx
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [inputMode, setInputMode] = useState('draw');
  const [sketchBlob, setSketchBlob] = useState(null);
  const [status, setStatus] = useState({ message: '', isError: false });

  // Extrusion State
  const [extrusionDepth, setExtrusionDepth] = useState(0.25);
  const [extrusionSmoothSteps, setExtrusionSmoothSteps] = useState(0);
  const [sketchThickness, setSketchThickness] = useState(0);

  // Revolution State
  const [revolutionSegments, setRevolutionSegments] = useState(64);
  const [revolutionCapBottom, setRevolutionCapBottom] = useState(false);
  const [revolutionCapTop, setRevolutionCapTop] = useState(false);
  const [revolutionAxisOffsetX, setRevolutionAxisOffsetX] = useState(0);
  const [revolutionHollow, setRevolutionHollow] = useState(false);
  const [revolutionWallThickness, setRevolutionWallThickness] = useState(0.05);
  const [revolutionAngleDegrees, setRevolutionAngleDegrees] = useState(360);

  // HeightMap State
  const [heightScale, setHeightScale] = useState(0.35);
  const [heightWithBase, setHeightWithBase] = useState(true);
  const [heightBlurSigma, setHeightBlurSigma] = useState(0);
  const [heightResolution, setHeightResolution] = useState(64);
  const [heightBulgeStrength, setHeightBulgeStrength] = useState(0);

  const [evaluationLog, setEvaluationLog] = useState([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState({
    completed: 0,
    total: 0,
    label: ''
  });
  const [batchLogs, setBatchLogs] = useState([]);

  const updatePreviewUrl = (nextUrl) => {
    setPreviewUrl((prev) => {
      if (prev) {
        URL.revokeObjectURL(prev);
      }
      return nextUrl;
    });
  };

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem('sketch3d-evaluation-log');
      if (stored) {
        setEvaluationLog(JSON.parse(stored));
      }
    } catch (error) {
      // ignore storage errors
    }
  }, []);

  useEffect(() => {
    if (inputMode === 'draw') {
      setFile(null);
      updatePreviewUrl(null);
    }
  }, [inputMode]);

  const appendEvaluationRecord = (record) => {
    setEvaluationLog((prev) => {
      const next = [...prev, record];
      try {
        localStorage.setItem('sketch3d-evaluation-log', JSON.stringify(next));
      } catch (error) {
        // ignore storage writes
      }
      return next;
    });
  };

  const updateEntryResult = useCallback(
    (index, modeKey, updates) => {
      setEvaluationFiles((prev) =>
        prev.map((entry, idx) => {
          if (idx !== index) return entry;
          const baseResults = entry.results ? { ...entry.results } : createEmptyResults();
          const previous = baseResults[modeKey] || { status: 'pending' };
          baseResults[modeKey] = { ...previous, ...updates };
          return { ...entry, results: baseResults };
        })
      );
    },
    [setEvaluationFiles]
  );

  const runEvaluationForEntry = useCallback(
    async (entryIndex, overrideMode) => {
      const entry = evaluationFiles[entryIndex];
      if (!entry?.file) {
        return null;
      }
      const targetMode = overrideMode || mode;
      const storedSettings = entry.settings
        ? cloneSettings(entry.settings)
        : cloneSettings(getEvaluationSettings());
      const modeSettings = storedSettings[targetMode] || {};

      updateEntryResult(entryIndex, targetMode, {
        status: 'running',
        error: null,
        startedAt: new Date().toISOString()
      });

      const startTime = performance.now();
      try {
        const response = await uploadSketch(entry.file, targetMode, modeSettings);
        const durationMs = Math.round(performance.now() - startTime);
        updateEntryResult(entryIndex, targetMode, {
          status: 'done',
          resultMesh: response.mesh,
          resultImage: response.image,
          token: response.token,
          durationMs,
          completedAt: new Date().toISOString(),
          settings: modeSettings
        });
        setEvaluationFiles((prev) =>
          prev.map((item, idx) =>
            idx === entryIndex ? { ...item, settings: storedSettings } : item
          )
        );
        return {
          success: true,
          fileName: entry.file.name,
          mode: targetMode,
          durationMs,
          mesh: response.mesh,
          image: response.image,
          token: response.token
        };
      } catch (error) {
        console.error(error);
        updateEntryResult(entryIndex, targetMode, {
          status: 'error',
          error: error?.message || 'Processing failed'
        });
        return {
          success: false,
          fileName: entry.file.name,
          mode: targetMode,
          error: error?.message || 'Processing failed'
        };
      }
    },
    [evaluationFiles, mode, updateEntryResult, setEvaluationFiles] // getEvaluationSettings is defined later; avoid reference cycle
  );

  const handleBatchRun = useCallback(async () => {
    if (batchRunning || evaluationFiles.length === 0) return;
    const totalSteps = evaluationFiles.length * RUN_MODES.length;
    if (totalSteps === 0) return;
    setBatchRunning(true);
    setBatchLogs([]);
    setBatchProgress({
      completed: 0,
      total: totalSteps,
      label: 'Starting batch evaluation...'
    });

    let completed = 0;
    for (let fileIdx = 0; fileIdx < evaluationFiles.length; fileIdx += 1) {
      const entry = evaluationFiles[fileIdx];
      const fileLabel = entry?.file?.name || `Sketch ${fileIdx + 1}`;
      for (const modeKey of RUN_MODES) {
        setBatchProgress({
          completed,
          total: totalSteps,
          label: `Processing ${fileLabel} (${modeKey})`
        });
        const record =
          (await runEvaluationForEntry(fileIdx, modeKey)) ||
          {
            success: false,
            fileName: fileLabel,
            mode: modeKey,
            error: 'Skipped'
          };
        completed += 1;
        setBatchLogs((prev) => [...prev, record]);
        setBatchProgress({
          completed,
          total: totalSteps,
          label: `Completed ${fileLabel} (${modeKey})`
        });
      }
    }
    setBatchProgress({
      completed: totalSteps,
      total: totalSteps,
      label: 'Batch evaluation finished'
    });
    setBatchRunning(false);
  }, [batchRunning, evaluationFiles, runEvaluationForEntry]);

  const downloadCsvReport = useCallback(() => {
    const rows = [
      ['file', 'mode', 'status', 'image', 'mesh', 'token', 'duration_ms', 'error']
    ];
    evaluationFiles.forEach((entry) => {
      RUN_MODES.forEach((modeKey) => {
        const res = entry.results?.[modeKey];
        if (!res) return;
        rows.push([
          entry.file?.name || '',
          modeKey,
          res.status || 'pending',
          res.resultImage || '',
          res.resultMesh || '',
          res.token || '',
          res.durationMs ?? '',
          res.error || ''
        ]);
      });
    });
    if (rows.length <= 1) {
      return;
    }
    const csvText = rows
      .map((row) =>
        row
          .map((value) => `"${String(value ?? '').replace(/"/g, '""')}"`)
          .join(',')
      )
      .join('\n');
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `sketch3d-evaluation-${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [evaluationFiles]);

  // Load settings when selection changes
  useEffect(() => {
    if (evaluationMode && evaluationFiles[selectedEvalIndex]) {
      const savedSettings = evaluationFiles[selectedEvalIndex].settings;
      if (!savedSettings) return;

      // Load Revolution settings
      if (savedSettings.revolution) {
        setRevolutionSegments(savedSettings.revolution.revolutionSegments ?? 64);
        setRevolutionCapBottom(savedSettings.revolution.revolutionCapBottom ?? false);
        setRevolutionCapTop(savedSettings.revolution.revolutionCapTop ?? false);
        setRevolutionAxisOffsetX(savedSettings.revolution.revolutionAxisOffsetX ?? 0);
        setRevolutionHollow(savedSettings.revolution.revolutionHollow ?? false);
        setRevolutionWallThickness(savedSettings.revolution.revolutionWallThickness ?? 0.05);
        setRevolutionAngleDegrees(savedSettings.revolution.revolutionAngleDegrees ?? 360);
      }

      // Load Extrusion settings
      if (savedSettings.extrusion) {
        setExtrusionDepth(savedSettings.extrusion.extrusionDepth ?? 0.25);
        setExtrusionSmoothSteps(savedSettings.extrusion.extrusionSmoothSteps ?? 0);
        setSketchThickness(savedSettings.extrusion.sketchThickness ?? 0);
      }

      // Load Heightmap settings
      if (savedSettings.heightmap) {
        setHeightScale(savedSettings.heightmap.heightScale ?? 0.35);
        setHeightWithBase(savedSettings.heightmap.heightWithBase ?? true);
        setHeightBlurSigma(savedSettings.heightmap.heightBlurSigma ?? 0);
        setHeightResolution(savedSettings.heightmap.heightResolution ?? 64);
        setHeightBulgeStrength(savedSettings.heightmap.heightBulgeStrength ?? 0);
      }
    }
  }, [selectedEvalIndex, evaluationMode]); // Intentionally omitting evaluationFiles to avoid loops

  const emitSettingsSnapshot = useCallback(() => {
    const currentSettings = {
      extrusion: {
        extrusionDepth,
        extrusionSmoothSteps,
        sketchThickness
      },
      revolution: {
        revolutionSegments,
        revolutionCapBottom,
        revolutionCapTop,
        revolutionAxisOffsetX,
        revolutionHollow,
        revolutionWallThickness,
        revolutionAngleDegrees
      },
      heightmap: {
        heightScale,
        heightWithBase,
        heightBlurSigma,
        heightResolution,
        heightBulgeStrength
      }
    };

    if (onSettingsChange) {
      onSettingsChange(currentSettings);
    }

    // Sync back to evaluation file if in evaluation mode
    if (evaluationMode && evaluationFiles[selectedEvalIndex]) {
      // Only update if actually different to avoid render loops if possible, 
      // but for now we rely on React state diffing.
      // We need to update the specific file entry in the array.
      // Note: We can't call setEvaluationFiles directly here if it triggers this effect again immediately.
      // But this effect depends on the primitive values (extrusionDepth, etc).
      // Updating evaluationFiles will trigger re-render of parent, passing down new evaluationFiles.
      // But the "Load" effect above depends only on selectedEvalIndex. So it won't re-run.

      setEvaluationFiles(prev => {
        const next = [...prev];
        // Check if changed to avoid unnecessary updates
        if (JSON.stringify(next[selectedEvalIndex].settings) !== JSON.stringify(currentSettings)) {
          next[selectedEvalIndex] = {
            ...next[selectedEvalIndex],
            settings: cloneSettings(currentSettings)
          };
          return next;
        }
        return prev;
      });
    }
  }, [
    onSettingsChange,
    evaluationMode,
    selectedEvalIndex,
    // evaluationFiles, // Omitting to avoid loop, we use functional update
    extrusionDepth,
    extrusionSmoothSteps,
    sketchThickness,
    revolutionSegments,
    revolutionCapBottom,
    revolutionCapTop,
    revolutionAxisOffsetX,
    revolutionHollow,
    revolutionWallThickness,
    revolutionAngleDegrees,
    heightScale,
    heightWithBase,
    heightBlurSigma,
    heightResolution,
    heightBulgeStrength
  ]);

  useEffect(() => {
    emitSettingsSnapshot();
  }, [emitSettingsSnapshot]);

  const getEvaluationSettings = useCallback(() => {
    return {
      extrusion: {
        extrusionDepth,
        extrusionSmoothSteps,
        sketchThickness
      },
      revolution: {
        revolutionSegments,
        revolutionCapBottom,
        revolutionCapTop,
        revolutionAxisOffsetX,
        revolutionHollow,
        revolutionWallThickness,
        revolutionAngleDegrees
      },
      heightmap: {
        heightScale,
        heightWithBase,
        heightBlurSigma,
        heightResolution,
        heightBulgeStrength
      }
    };
  }, [
    extrusionDepth,
    extrusionSmoothSteps,
    sketchThickness,
    revolutionSegments,
    revolutionCapBottom,
    revolutionCapTop,
    revolutionAxisOffsetX,
    revolutionHollow,
    revolutionWallThickness,
    revolutionAngleDegrees,
    heightScale,
    heightWithBase,
    heightBlurSigma,
    heightResolution,
    heightBulgeStrength
  ]);

  const handleCanvasSave = (file) => {
    if (file) {
      setSketchBlob(file);
      setFile(null);
      updatePreviewUrl(null);
      setStatus({ message: 'Sketch saved. You can now submit it.', isError: false });
    } else {
      setSketchBlob(null);
    }
  };

  const handleFileSelection = async (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) {
      setFile(null);
      setSketchBlob(null);
      updatePreviewUrl(null);
      return;
    }
    setStatus({ message: '', isError: false });
    const lowerName = selectedFile.name?.toLowerCase() || '';
    const isSvg =
      selectedFile.type === 'image/svg+xml' || lowerName.endsWith('.svg');
    if (isSvg) {
      setFile(selectedFile);
      setSketchBlob(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      updatePreviewUrl(url);
      return;
    }
    try {
      setStatus({ message: 'Vectorizing raster sketch...', isError: false });
      const { svg } = await vectorizeRaster(selectedFile, sketchThickness);
      const blob = new Blob([svg], { type: 'image/svg+xml' });
      const fileName =
        (selectedFile.name?.replace(/\.[^.]+$/, '') || 'sketch') + '.svg';
      const vectorFile = new File([blob], fileName, {
        type: 'image/svg+xml'
      });
      setFile(vectorFile);
      setSketchBlob(vectorFile);
      const url = URL.createObjectURL(blob);
      updatePreviewUrl(url);
      setStatus({
        message: 'Raster sketch converted to SVG. Ready to submit.',
        isError: false
      });
    } catch (error) {
      console.error('[vectorize]', error);
      setStatus({
        message: 'Vectorization failed. Please try another sketch.',
        isError: true
      });
      setFile(null);
      setSketchBlob(null);
      updatePreviewUrl(null);
    }
  };

  const handleSubmit = async () => {
    setStatus({ message: '', isError: false });
    if (inputMode === 'upload' && !file) {
      setStatus({ message: 'Please select an image before generating.', isError: true });
      return;
    }
    if (!sketchBlob) {
      setStatus({ message: 'Please provide a sketch before generating.', isError: true });
      return;
    }
    const payload = sketchBlob;
    try {
      let settings = {};
      if (mode === 'extrusion') {
        settings = {
          extrusionDepth,
          extrusionSmoothSteps,
          sketchThickness,
        };
      } else if (mode === 'revolution') {
        settings = {
          revolutionSegments,
          revolutionCapBottom,
          revolutionCapTop,
          revolutionAxisOffsetX,
          revolutionHollow,
          revolutionWallThickness,
          revolutionAngleDegrees,
        };
      } else if (mode === 'heightmap') {
        settings = {
          heightScale,
          heightWithBase,
          heightBlurSigma,
          heightResolution,
          heightBulgeStrength,
        };
      }
      setStatus({ message: 'Submitting to the backend, waiting for render result...', isError: false });
      await uploadSketch(payload, mode, settings);
      setStatus({ message: 'Sketch submitted successfully. The preview will update shortly.', isError: false });
      if (evaluationMode) {
        const submittedName = payload.name || (inputMode === 'draw' ? 'canvas.svg' : 'upload.svg');
        const record = {
          timestamp: new Date().toISOString(),
          mode,
          inputMode,
          fileName: submittedName,
          settings: { ...settings }
        };
        appendEvaluationRecord(record);
      }
      window.dispatchEvent(new Event('sketch-refresh'));
    } catch (error) {
      setStatus({ message: 'Upload failed. Please try again later.', isError: true });
    }
  };

  const totalBatchJobs = evaluationFiles.length * RUN_MODES.length;
  const completedBatchJobs = evaluationFiles.reduce((sum, entry) => {
    if (!entry?.results) return sum;
    return (
      sum +
      RUN_MODES.reduce(
        (inner, modeKey) => inner + (entry.results[modeKey]?.status === 'done' ? 1 : 0),
        0
      )
    );
  }, 0);
  const hasResultData = completedBatchJobs > 0;

  return (
    <Container>
      <Navbar
        evaluationMode={evaluationMode}
        setEvaluationMode={setEvaluationMode}
      />

      <Panel>
        <HeaderRow>
          <Title>Generation Controls</Title>
        </HeaderRow>

        {(!evaluationMode || evaluationFiles.length > 0) && (
          <SegmentedControl
            value={mode}
            onChange={setMode}
            options={[
              { label: 'Height Map', value: 'heightmap' },
              { label: 'Extrusion', value: 'extrusion' },
              { label: 'Revolution', value: 'revolution' }
            ]}
          />
        )}

        {!evaluationMode && (
          <SegmentedControl
            value={inputMode}
            onChange={setInputMode}
            options={[
              { label: 'Draw on Canvas', value: 'draw' },
              { label: 'Upload Image', value: 'upload' }
            ]}
          />
        )}

        {!evaluationMode && (
          <>
            {inputMode === 'upload' ? (
              <ImageUpload
                file={file}
                previewUrl={previewUrl}
                onFileSelect={handleFileSelection}
                showAxis={mode === 'revolution'}
                axisOffsetX={revolutionAxisOffsetX}
              />
            ) : (
              <SketchCanvas
                onSketchSaved={handleCanvasSave}
                showRevolutionAxis={mode === 'revolution'}
                axisOffsetX={revolutionAxisOffsetX}
              />
            )}

            {mode === 'extrusion' && (
              <ExtrusionControls
                depth={extrusionDepth} setDepth={setExtrusionDepth}
                smoothSteps={extrusionSmoothSteps} setSmoothSteps={setExtrusionSmoothSteps}
                thickness={sketchThickness} setThickness={setSketchThickness}
              />
            )}

            {mode === 'revolution' && (
              <RevolutionControls
                segments={revolutionSegments} setSegments={setRevolutionSegments}
                axisOffsetX={revolutionAxisOffsetX} setAxisOffsetX={setRevolutionAxisOffsetX}
                angle={revolutionAngleDegrees} setAngle={setRevolutionAngleDegrees}
                capBottom={revolutionCapBottom} setCapBottom={setRevolutionCapBottom}
                capTop={revolutionCapTop} setCapTop={setRevolutionCapTop}
                hollow={revolutionHollow} setHollow={setRevolutionHollow}
                wallThickness={revolutionWallThickness} setWallThickness={setRevolutionWallThickness}
              />
            )}

            {mode === 'heightmap' && (
              <HeightMapControls
                scale={heightScale} setScale={setHeightScale}
                blurSigma={heightBlurSigma} setBlurSigma={setHeightBlurSigma}
                resolution={heightResolution} setResolution={setHeightResolution}
                bulge={heightBulgeStrength} setBulge={setHeightBulgeStrength}
                withBase={heightWithBase} setWithBase={setHeightWithBase}
              />
            )}

            <Button onClick={handleSubmit}>
              <Upload size={18} />
              Generate
            </Button>
            {status.message && <StatusMessage $isError={status.isError}>{status.message}</StatusMessage>}
          </>
        )}

        {evaluationMode && (
          <>
            <div style={{ marginBottom: 16 }}>
              <label htmlFor="eval-upload" style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                padding: '12px',
                background: '#27272a',
                borderRadius: '8px',
                cursor: 'pointer',
                border: '1px dashed #3f3f46',
                color: '#a1a1aa',
                fontWeight: 600
              }}>
                <Upload size={18} />
                Add Sketches
              </label>
              <input
                id="eval-upload"
                type="file"
                multiple
                accept="image/*,.svg"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  const template = getEvaluationSettings();
                  const newEntries = files.map(f => ({
                    file: f,
                    previewUrl: URL.createObjectURL(f),
                    results: createEmptyResults(),
                    settings: cloneSettings(template)
                  }));
                  setEvaluationFiles(prev => [...prev, ...newEntries]);
                }}
              />
            </div>



            {evaluationFiles[selectedEvalIndex] ? (
              <>
                <div style={{ marginBottom: 12, padding: 8, background: '#27272a', borderRadius: 4, fontSize: '0.85rem', color: '#fff' }}>
                  Editing: <strong>{evaluationFiles[selectedEvalIndex].file.name}</strong>
                </div>

                {mode === 'extrusion' && (
                  <ExtrusionControls
                    depth={extrusionDepth} setDepth={setExtrusionDepth}
                    smoothSteps={extrusionSmoothSteps} setSmoothSteps={setExtrusionSmoothSteps}
                    thickness={sketchThickness} setThickness={setSketchThickness}
                  />
                )}

                {mode === 'revolution' && (
                  <RevolutionControls
                    segments={revolutionSegments} setSegments={setRevolutionSegments}
                    axisOffsetX={revolutionAxisOffsetX} setAxisOffsetX={setRevolutionAxisOffsetX}
                    angle={revolutionAngleDegrees} setAngle={setRevolutionAngleDegrees}
                    capBottom={revolutionCapBottom} setCapBottom={setRevolutionCapBottom}
                    capTop={revolutionCapTop} setCapTop={setRevolutionCapTop}
                    hollow={revolutionHollow} setHollow={setRevolutionHollow}
                    wallThickness={revolutionWallThickness} setWallThickness={setRevolutionWallThickness}
                  />
                )}

                {mode === 'heightmap' && (
                  <HeightMapControls
                    scale={heightScale} setScale={setHeightScale}
                    blurSigma={heightBlurSigma} setBlurSigma={setHeightBlurSigma}
                    resolution={heightResolution} setResolution={setHeightResolution}
                    bulge={heightBulgeStrength} setBulge={setHeightBulgeStrength}
                    withBase={heightWithBase} setWithBase={setHeightWithBase}
                  />
                )}

                <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {batchProgress.label && (
                    <div style={{ fontSize: '0.8rem', color: '#a1a1aa', textAlign: 'center' }}>
                      {batchProgress.label} ({completedBatchJobs}/{totalBatchJobs})
                    </div>
                  )}

                  <Button
                    onClick={handleBatchRun}
                    disabled={batchRunning || evaluationFiles.length === 0}
                    style={{ opacity: batchRunning ? 0.7 : 1 }}
                  >
                    <Upload size={18} />
                    {batchRunning ? 'Running Batch...' : 'Run All (3 Modes)'}
                  </Button>

                  <SecondaryButton
                    type="button"
                    disabled={!hasResultData}
                    onClick={downloadCsvReport}
                    style={{ width: '100%', justifyContent: 'center', display: 'flex' }}
                  >
                    Download CSV Report
                  </SecondaryButton>
                </div>
              </>
            ) : (
              <div style={{ color: '#666', textAlign: 'center', marginTop: 20 }}>
                Select a file to edit settings
              </div>
            )}
          </>
        )}
      </Panel>
    </Container>
  );
}

export default ControlsPanel;
