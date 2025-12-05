import { useMemo, useState } from 'react';
import styled from 'styled-components';
import Viewer3D from './Viewer3D';
import { uploadSketch } from '../services/api';

const Wrapper = styled.div`
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 16px;
  height: 100%;
`;

const Panel = styled.div`
  background: ${(props) => props.theme.colors.surface};
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: ${(props) => props.theme.radii.xl};
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
`;

const Title = styled.h3`
  margin: 0;
  font-size: 1rem;
  font-weight: 700;
  color: ${(props) => props.theme.colors.text};
`;

const FileInput = styled.input`
  width: 100%;
`;

const ModeRow = styled.div`
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
`;

const CheckboxLabel = styled.label`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: ${(props) => props.theme.colors.surfaceHighlight};
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: ${(props) => props.theme.radii.sm};
  padding: 6px 10px;
  cursor: pointer;
`;

const RunButton = styled.button`
  padding: 12px;
  border: none;
  border-radius: ${(props) => props.theme.radii.lg};
  background: ${(props) => props.theme.colors.accent};
  color: #000;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const JobsList = styled.div`
  flex: 1;
  overflow-y: auto;
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: ${(props) => props.theme.radii.lg};
  background: ${(props) => props.theme.colors.surfaceHighlight};
`;

const JobRow = styled.div`
  padding: 10px 12px;
  border-bottom: 1px solid ${(props) => props.theme.colors.border};
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
`;

const StatusTag = styled.span`
  font-size: 0.8rem;
  padding: 2px 8px;
  border-radius: ${(props) => props.theme.radii.md};
  background: ${(props) =>
    props.$status === 'done'
      ? 'rgba(52,211,153,0.15)'
      : props.$status === 'running'
      ? 'rgba(94,234,212,0.15)'
      : props.$status === 'error'
      ? 'rgba(248,113,113,0.15)'
      : 'rgba(156,163,175,0.15)'};
  color: ${(props) =>
    props.$status === 'done'
      ? '#10b981'
      : props.$status === 'running'
      ? '#0ea5e9'
      : props.$status === 'error'
      ? '#ef4444'
      : '#6b7280'};
`;

const Actions = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
`;

const SmallButton = styled.button`
  padding: 6px 8px;
  border-radius: ${(props) => props.theme.radii.sm};
  border: 1px solid ${(props) => props.theme.colors.border};
  background: ${(props) => props.theme.colors.surface};
  color: ${(props) => props.theme.colors.text};
  cursor: pointer;
  font-size: 0.85rem;
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ViewerWrapper = styled.div`
  background: ${(props) => props.theme.colors.surface};
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: ${(props) => props.theme.radii.xl};
  padding: 12px;
  min-height: 420px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Notice = styled.div`
  font-size: 0.9rem;
  color: ${(props) => props.theme.colors.textSecondary};
`;

function buildModeSettings(settings) {
  const extrusion = settings?.extrusion || {};
  const revolution = settings?.revolution || {};
  const heightmap = settings?.heightmap || {};
  return {
    extrusion: {
      extrusionDepth: extrusion.extrusionDepth ?? 0.25,
      extrusionSmoothSteps: extrusion.extrusionSmoothSteps ?? 0,
      sketchThickness: extrusion.sketchThickness ?? 0
    },
    revolution: {
      revolutionSegments: revolution.revolutionSegments ?? 64,
      revolutionCapBottom: revolution.revolutionCapBottom ?? false,
      revolutionCapTop: revolution.revolutionCapTop ?? false,
      revolutionAxisOffsetX: revolution.revolutionAxisOffsetX ?? 0,
      revolutionHollow: revolution.revolutionHollow ?? false,
      revolutionWallThickness: revolution.revolutionWallThickness ?? 0.05,
      revolutionAngleDegrees: revolution.revolutionAngleDegrees ?? 360
    },
    heightmap: {
      heightScale: heightmap.heightScale ?? 0.35,
      heightWithBase: heightmap.heightWithBase ?? true,
      heightBlurSigma: heightmap.heightBlurSigma ?? 0,
      heightResolution: heightmap.heightResolution ?? 64,
      heightBulgeStrength: heightmap.heightBulgeStrength ?? 0
    }
  };
}

export default function EvaluationPanel({ settings }) {
  const [files, setFiles] = useState([]);
  const [selectedModes, setSelectedModes] = useState({
    extrusion: true,
    revolution: true,
    heightmap: true
  });
  const [jobs, setJobs] = useState([]);
  const [running, setRunning] = useState(false);
  const [selectedMesh, setSelectedMesh] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  const modeSettings = useMemo(() => buildModeSettings(settings), [settings]);

  const toggleMode = (mode) => {
    setSelectedModes((prev) => ({ ...prev, [mode]: !prev[mode] }));
  };

  const handleFiles = (event) => {
    const picked = Array.from(event.target.files || []);
    setFiles(picked);
  };

  const runBatch = async () => {
    if (!files.length) return;
    const enabledModes = Object.entries(selectedModes).filter(([, v]) => v);
    if (!enabledModes.length) return;

    const queue = [];
    files.forEach((file) => {
      enabledModes.forEach(([mode]) => {
        queue.push({
          id: `${file.name}-${mode}-${Math.random().toString(36).slice(2)}`,
          file,
          name: file.name,
          mode,
          status: 'queued'
        });
      });
    });
    setJobs(queue);
    setRunning(true);
    setSelectedMesh(null);
    setSelectedImage(null);

    const nextJobs = [...queue];
    for (let i = 0; i < nextJobs.length; i += 1) {
      const job = nextJobs[i];
      const settingsForMode = modeSettings[job.mode] || {};
      setJobs((prev) =>
        prev.map((j) =>
          j.id === job.id ? { ...j, status: 'running', startedAt: Date.now() } : j
        )
      );
      const start = performance.now();
      try {
        const result = await uploadSketch(job.file, job.mode, settingsForMode);
        const durationMs = performance.now() - start;
        setJobs((prev) =>
          prev.map((j) =>
            j.id === job.id
              ? {
                  ...j,
                  status: 'done',
                  durationMs,
                  image: result?.image || null,
                  mesh: result?.mesh || null,
                  token: result?.token || null
                }
              : j
          )
        );
        if (!selectedMesh && result?.mesh) {
          setSelectedMesh(result.mesh);
        }
        if (!selectedImage && result?.image) {
          setSelectedImage(result.image);
        }
      } catch (error) {
        const durationMs = performance.now() - start;
        setJobs((prev) =>
          prev.map((j) =>
            j.id === job.id
              ? { ...j, status: 'error', error: error.message, durationMs }
              : j
          )
        );
      }
    }
    setRunning(false);
  };

  const doneCount = jobs.filter((j) => j.status === 'done').length;

  return (
    <Wrapper>
      <Panel>
        <Title>Evaluation Runner</Title>
        <Notice>批量上传素描，按当前参数对选定算法依次运行，并记录耗时。</Notice>
        <div>
          <strong>1) 选择文件（多选）：</strong>
          <FileInput
            type="file"
            multiple
            accept="image/*,.svg"
            onChange={handleFiles}
          />
          <Notice>{files.length} files selected.</Notice>
        </div>
        <div>
          <strong>2) 选择要运行的算法：</strong>
          <ModeRow>
            {['extrusion', 'revolution', 'heightmap'].map((mode) => (
              <CheckboxLabel key={mode}>
                <input
                  type="checkbox"
                  checked={!!selectedModes[mode]}
                  onChange={() => toggleMode(mode)}
                />
                {mode}
              </CheckboxLabel>
            ))}
          </ModeRow>
        </div>
        <RunButton onClick={runBatch} disabled={running || !files.length}>
          {running ? 'Running...' : 'Run Batch'}
        </RunButton>

        <JobsList>
          {jobs.length === 0 && (
            <JobRow>
              <span>等待开始。选择文件并点击 Run Batch。</span>
            </JobRow>
          )}
          {jobs.map((job) => (
            <JobRow key={job.id}>
              <div>
                <div style={{ fontWeight: 700 }}>{job.name}</div>
                <div style={{ fontSize: '0.85rem', color: '#9ca3af' }}>
                  {job.mode} ·{' '}
                  {job.durationMs
                    ? `${job.durationMs.toFixed(0)} ms`
                    : job.status === 'running'
                    ? 'running...'
                    : 'pending'}
                </div>
                {job.error && (
                  <div style={{ color: '#ef4444', fontSize: '0.85rem' }}>
                    {job.error}
                  </div>
                )}
              </div>
              <Actions>
                {job.image && (
                  <SmallButton onClick={() => setSelectedImage(job.image)}>
                    Image
                  </SmallButton>
                )}
                {job.mesh && (
                  <SmallButton onClick={() => setSelectedMesh(job.mesh)}>
                    3D
                  </SmallButton>
                )}
                <StatusTag $status={job.status}>{job.status}</StatusTag>
              </Actions>
            </JobRow>
          ))}
        </JobsList>
        <Notice>
          {doneCount}/{jobs.length} 完成。
        </Notice>
      </Panel>

      <ViewerWrapper>
        <Title>Result Viewer</Title>
        {!selectedMesh && !selectedImage && (
          <Notice>选择一个条目的 Image 或 3D 查看结果。</Notice>
        )}
        {selectedMesh && (
          <div style={{ flex: 1, minHeight: 320 }}>
            <Viewer3D meshUrl={selectedMesh} />
          </div>
        )}
        {selectedImage && (
          <div style={{ border: '1px solid #222', borderRadius: 8, overflow: 'hidden' }}>
            <img
              src={selectedImage}
              alt="render"
              style={{ width: '100%', display: 'block' }}
            />
          </div>
        )}
      </ViewerWrapper>
    </Wrapper>
  );
}

