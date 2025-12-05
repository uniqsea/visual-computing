import styled from 'styled-components';
import SliderControl from './SliderControl';
import CheckboxControl from './CheckboxControl';

const FieldGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 12px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: ${(props) => props.theme.radii.md};
`;

export default function HeightMapControls({
    scale, setScale,
    blurSigma, setBlurSigma,
    resolution, setResolution,
    bulge, setBulge,
    withBase, setWithBase
}) {
    return (
        <FieldGroup>
            <SliderControl
                label="Height Scale"
                description="Vertical scaling factor for the height map."
                value={scale}
                onChange={setScale}
                min={0.1}
                max={2.0}
                step={0.05}
                formatValue={(v) => v.toFixed(2)}
            />
            <SliderControl
                label="Blur Sigma"
                description="Gaussian blur radius to smooth the height map."
                value={blurSigma}
                onChange={setBlurSigma}
                min={0}
                max={10}
                step={0.5}
                formatValue={(v) => v.toFixed(2)}
            />
            <SliderControl
                label="Resolution"
                description="Grid resolution for the height map mesh."
                value={resolution}
                onChange={setResolution}
                min={32}
                max={256}
                step={32}
            />
            <SliderControl
                label="Bulge Strength"
                description="Applies a spherical bulge effect to the height map."
                value={bulge}
                onChange={setBulge}
                min={0}
                max={1}
                step={0.1}
                formatValue={(v) => v.toFixed(1)}
            />

            <CheckboxControl
                label="Add Base"
                description="Adds a solid base below the height map."
                checked={withBase}
                onChange={setWithBase}
            />
        </FieldGroup>
    );
}
