import { useCallback, useEffect, useMemo, useState } from 'react';
import styled, { keyframes } from 'styled-components';
import { fetchLatestRender } from '../services/api';
import Viewer3D from './Viewer3D';
import HintBanner from './controls/HintBanner';

const fadeIn = keyframes`
  from { opacity: 0; }
  to { opacity: 1; }
`;

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  animation: ${fadeIn} 0.5s ease-in-out;
  background: ${(props) => props.theme.colors.surface};
  padding: 0;
  border-radius: ${(props) => props.theme.radii.xl};
  height: 100%;
  border: 1px solid ${(props) => props.theme.colors.border};
  overflow: hidden;
  position: relative;
`;

const PreviewImage = styled.img`
  width: 100%;
  border-radius: 8px;
  border: 1px solid ${(props) => props.theme.colors.border};
`;

const Placeholder = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
  background: ${(props) => props.theme.colors.background};
  color: ${(props) => props.theme.colors.textSecondary};
  gap: 24px;
`;

const PlaceholderIcon = styled.div`
  width: 64px;
  height: 64px;
  color: ${(props) => props.theme.colors.accent};
  opacity: 0.8;
  
  svg {
    width: 100%;
    height: 100%;
  }
`;

const PlaceholderText = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  
  h3 {
    font-size: 1.5rem;
    color: ${(props) => props.theme.colors.text};
    font-weight: 600;
  }
  
  p {
    font-size: 0.9rem;
    color: ${(props) => props.theme.colors.textSecondary};
  }
`;

const SplitContainer = styled.div`
  display: flex;
  height: 100%;
  width: 100%;
  overflow: hidden;
`;

const ThumbnailList = styled.div`
  width: 300px;
  background: ${(props) => props.theme.colors.surfaceHighlight};
  border-right: 1px solid ${(props) => props.theme.colors.border};
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 12px;
  overflow-y: auto;
  flex-shrink: 0;
`;

const ThumbnailContainer = styled.div`
  width: 100%;
  aspect-ratio: 1;
  border-radius: ${(props) => props.theme.radii.md};
  border: 2px solid ${(props) => (props.$active ? props.theme.colors.accent : 'transparent')};
  background: ${(props) => props.theme.colors.surface};
  position: relative;
  overflow: hidden;
  cursor: pointer;
  opacity: ${(props) => (props.$active ? 1 : 0.6)};
  transition: all 0.2s;

  &:hover {
    opacity: 1;
    border-color: ${(props) => (props.$active ? props.theme.colors.accent : props.theme.colors.border)};
  }
`;

const ThumbnailImage = styled.div`
  width: 100%;
  height: 100%;
  background-image: url(${(props) => props.$src});
  background-size: cover;
  background-position: center;
`;

const MainView = styled.div`
  flex: 1;
  position: relative;
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #000;
`;

function RenderPreview({ mode, evaluationMode, evaluationFiles, selectedEvalIndex, setSelectedEvalIndex }) {
  const [imageUrl, setImageUrl] = useState(null);
  const [meshUrl, setMeshUrl] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [showRevolutionHint, setShowRevolutionHint] = useState(true);

  // Interactive Mode Logic
  useEffect(() => {
    if (evaluationMode) return; // Skip in eval mode

    const load = () => {
      setLoading(true);
      fetchLatestRender()
        .then((result) => {
          setImageUrl(result.image);
          setMeshUrl(result.mesh);
          setError('');
        })
        .catch(() => {
          setImageUrl(null);
          setMeshUrl(null);
          setError('No render results available yet. Submit a sketch to get started.');
        })
        .finally(() => {
          setLoading(false);
        });
    };
    load();
    const handler = () => load();
    window.addEventListener('sketch-refresh', handler);
    return () => window.removeEventListener('sketch-refresh', handler);
  }, [evaluationMode]);

  const viewerProps = useMemo(
    () =>
      mode === 'heightmap'
        ? { cameraPosition: [0, 2.5, 0.01], target: [0, 0, 0] }
        : { cameraPosition: [0, 0, 2], target: [0, 0, 0] },
    [mode]
  );

  const saveComparison = useCallback(async () => {
    if (!evaluationMode) return;
    const selectedFile = evaluationFiles[selectedEvalIndex];
    if (!selectedFile?.previewUrl || !selectedFile?.resultImage) return;
    const loadImage = (src) =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.crossOrigin = 'anonymous';
        img.src = src;
      });
    try {
      const [inputImg, outputImg] = await Promise.all([
        loadImage(selectedFile.previewUrl),
        loadImage(selectedFile.resultImage)
      ]);
      const width = Math.max(inputImg.width, outputImg.width);
      const height = inputImg.height + outputImg.height;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(inputImg, 0, 0, width, inputImg.height);
      ctx.drawImage(outputImg, 0, inputImg.height, width, outputImg.height);
      ctx.fillStyle = '#000';
      ctx.font = '16px sans-serif';
      ctx.fillText('Input', 8, 20);
      ctx.fillText('Result', 8, inputImg.height + 20);
      const link = document.createElement('a');
      link.download = `${selectedFile.file?.name || 'comparison'}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (err) {
      console.error('Failed to save comparison', err);
    }
  }, [evaluationMode, evaluationFiles, selectedEvalIndex]);

  if (evaluationMode) {
    const selectedFile = evaluationFiles[selectedEvalIndex];

    return (
      <Wrapper>
        <SplitContainer>
          <ThumbnailList>
            {evaluationFiles.map((file, idx) => {
              const settings = file.settings?.revolution || {};
              const axisOffsetX = settings.revolutionAxisOffsetX || 0;
              const axisPercent = Math.max(0, Math.min(1, 0.5 + axisOffsetX));
              const isActive = idx === selectedEvalIndex;

              return (
                <ThumbnailContainer
                  key={idx}
                  $active={isActive}
                  onClick={() => setSelectedEvalIndex(idx)}
                >
                  <ThumbnailImage $src={file.previewUrl} />
                  {mode === 'revolution' && isActive && (
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${axisPercent * 100}%`,
                        width: '2px',
                        background: 'rgba(180,180,180,0.8)',
                        transform: 'translateX(-1px)',
                        pointerEvents: 'none',
                        zIndex: 5
                      }}
                    />
                  )}
                </ThumbnailContainer>
              );
            })}
            {evaluationFiles.length === 0 && (
              <div style={{ padding: 10, textAlign: 'center', fontSize: '0.8rem', color: '#666' }}>
                No files
              </div>
            )}
          </ThumbnailList>
          <MainView>
            {selectedFile ? (
              <>
                {selectedFile.status === 'running' && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', color: 'white', zIndex: 20 }}>
                    Processing...
                  </div>
                )}

                {selectedFile.resultMesh ? (
                  <Viewer3D meshUrl={selectedFile.resultMesh} {...viewerProps} />
                ) : selectedFile.resultImage ? (
                  <PreviewImage src={selectedFile.resultImage} alt="Result" />
                ) : (
                  <div style={{ color: '#666', fontSize: '0.9rem' }}>No result generated yet</div>
                )}
                {selectedFile.resultImage && (
                  <div style={{ position: 'absolute', top: 12, right: 12, zIndex: 15 }}>
                    <button
                      type="button"
                      onClick={saveComparison}
                      style={{
                        padding: '8px 10px',
                        borderRadius: 6,
                        border: '1px solid #3f3f46',
                        background: '#1f2937',
                        color: '#e5e7eb',
                        cursor: 'pointer'
                      }}
                    >
                      Save input+result
                    </button>
                  </div>
                )}
              </>
            ) : (
              <Placeholder>Select or upload a file to start evaluation</Placeholder>
            )}
          </MainView>
        </SplitContainer>
      </Wrapper>
    );
  }

  // Interactive Mode Render
  if (loading) {
    return <Placeholder>Loading latest render...</Placeholder>;
  }

  return (
    <Wrapper>
      {mode === 'revolution' && showRevolutionHint && (
        <div style={{ position: 'absolute', top: 16, left: 16, zIndex: 10, maxWidth: 400 }}>
          <HintBanner onDismiss={() => setShowRevolutionHint(false)}>
            Revolution uses only the left side of your sketch relative to the axis line. Draw the profile on that half to control the lathe.
          </HintBanner>
        </div>
      )}
              {error && <Placeholder>{error}</Placeholder>}
      {!error && meshUrl ? (
                <Viewer3D meshUrl={meshUrl} {...viewerProps} />
      ) : (
        imageUrl && <PreviewImage src={imageUrl} alt="Rendered mesh" />
      )}
    </Wrapper>
  );
}

export default RenderPreview;
