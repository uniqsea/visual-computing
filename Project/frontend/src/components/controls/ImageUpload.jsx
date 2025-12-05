import styled from 'styled-components';
import { Upload } from 'lucide-react';

const FileInputContainer = styled.div`
  position: relative;
  padding: 0;
  border-radius: ${(props) => props.theme.radii.lg};
  border: 1px solid ${(props) => props.theme.colors.border};
  background: ${(props) => props.theme.colors.surface};
  color: ${(props) => props.theme.colors.textSecondary};
  cursor: pointer;
  text-align: center;
  transition: all 0.2s;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 320px;
  min-height: 320px;
  flex-shrink: 0;
  overflow: hidden;

  &:hover {
    border-color: ${(props) => props.theme.colors.textSecondary};
  }

  input {
    opacity: 0;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    cursor: pointer;
    z-index: 2;
  }
`;

export default function ImageUpload({
    file,
    previewUrl,
    onFileSelect,
    showAxis = false,
    axisOffsetX = 0
}) {
    const axisPercent = Math.max(0, Math.min(1, 0.5 + (axisOffsetX || 0)));
    return (
        <FileInputContainer>
            {previewUrl ? (
                <div
                    style={{
                        width: '100%',
                        height: '100%',
                        position: 'relative',
                        background: '#ffffff'
                    }}
                >
                    <img
                        src={previewUrl}
                        alt="Preview"
                        style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            padding: '16px'
                        }}
                    />
                    {showAxis && (
                        <div
                            style={{
                                position: 'absolute',
                                top: 0,
                                bottom: 0,
                                left: `${axisPercent * 100}%`,
                                width: '2px',
                                background: 'rgba(180,180,180,0.8)',
                                transform: 'translateX(-1px)',
                                pointerEvents: 'none'
                            }}
                        />
                    )}
                    <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        padding: '8px',
                        background: 'rgba(0,0,0,0.7)',
                        color: 'white',
                        fontSize: '0.8rem',
                        textAlign: 'center'
                    }}>
                        {file?.name || 'vectorized.svg'}
                    </div>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                    <Upload size={32} />
                    <span>Click to upload</span>
                </div>
            )}
            <input
                type="file"
                accept="image/*,.svg"
                onChange={onFileSelect}
            />
        </FileInputContainer>
    );
}
